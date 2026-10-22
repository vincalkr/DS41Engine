// tools/v41/engine/ds41_conv_disk.hpp - ds41_generate --serve: conversations kept on disk (--conversation-disk DIR), so a
// restarted engine resumes them instead of reading their whole prompt again.
//
// A record is one conversation at a turn boundary n: the state Ds41Dense::snapshot() took at the end of its prompt -
// every position-indexed stream's rows [0, n / ratio) (the compressed KV and indexer keys, the rows the conversation
// pool copies) and the snapshot rings (window K/V, ratio-2 compressor states) - plus its tokens, image keys and the
// boundary's last-row logits.  One directory a record:
//   meta.txt          written last (tmp + rename): version, layout fingerprint, n, rows per stream, regions file
//   tokens.bin, ident.bin (only with images), logits.bin
//   s<k>.bin          stream k's rows; the same conversation one turn later only writes the rows past the old ones
//   regions-<n>.bin   the rings at n (a new file per boundary, the old one removed after meta.txt points past it)
// So a crash at any point leaves the last complete boundary readable, and a record whose files do not match its
// meta.txt (or another model / cache layout) is skipped at start.
//
// Saves copy the device rows to host memory on the caller's thread (a few MB a turn: only the new rows and the ~6 MB
// of rings) and write the files on a background thread; restore() waits for that thread first.  Least recently used
// records leave when the directory passes its budget.
#pragma once

#include "ds41_dense.hpp"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace strata::v41 {

class Ds41ConvDisk {
public:
    ~Ds41ConvDisk() { stop(); }

    /// Open (create) dir and index the records there whose layout matches this engine.  budget: bytes on disk.
    bool init(const std::string & dir, size_t budget, const Ds41Dense & dense, const std::string & model_id,
              std::string & err) {
        namespace fs = std::filesystem;
        std::error_code aec;
        dir_ = fs::absolute(dir, aec).lexically_normal().string();   // the Monitor shows where the files are
        if (aec) dir_ = dir;
        budget_ = budget;
        dense.conv_tensors(streams_, regions_);
        if (streams_.empty() || regions_.empty()) { err = "no conversation state in this model"; return false; }
        region_bytes_ = 0;
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { for (int i = 0; i < 8; ++i) { h ^= (v >> (8 * i)) & 0xff; h *= 1099511628211ull; } };
        mix(kVersion);
        for (const auto & s : streams_) { mix(s.row_bytes); mix((uint64_t) s.ratio); }
        for (ggml_tensor * t : regions_) { mix(ggml_nbytes(t)); region_bytes_ += ggml_nbytes(t); }
        for (char c : model_id) mix((uint8_t) c);
        char fp[24];
        std::snprintf(fp, sizeof(fp), "%016llx", (unsigned long long) h);
        fp_ = fp;
        std::error_code ec;
        fs::create_directories(dir_, ec);
        if (ec) { err = "cannot create " + dir_ + ": " + ec.message(); return false; }
        for (const auto & d : fs::directory_iterator(dir_, ec)) {
            if (!d.is_directory() || d.path().filename().string().rfind("rec-", 0) != 0) continue;
            Entry e;
            if (load_entry(d.path().string(), e)) entries_.push_back(std::move(e));
            else fs::remove_all(d.path(), ec);   // incomplete, or another model / layout
        }
        on_ = true;
        writer_ = std::thread([this] { run(); });
        evict();
        return true;
    }
    bool enabled() const { return on_; }
    const std::string & dir() const { return dir_; }
    size_t budget() const { return budget_; }
    size_t records() const { std::lock_guard<std::mutex> lk(m_); return entries_.size(); }
    size_t bytes() const {
        std::lock_guard<std::mutex> lk(m_);
        size_t b = 0;
        for (const auto & e : entries_) b += e.bytes;
        return b;
    }

    /// Save the conversation whose state the snapshot tensors hold at boundary n (the engine is between requests).
    bool save(Ds41Dense & dense, const int * tokens, const uint64_t * ident, size_t ident_n, int64_t n,
              const float * logits, size_t n_logits, std::string & err) {
        if (!on_ || n < 1) return false;
        std::lock_guard<std::mutex> lk(m_);
        // the record this one continues (its tokens are a prefix of these): only the new rows are written
        int best = -1;
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].n <= n && prefix(entries_[i], tokens, ident, ident_n, n) &&
                (best < 0 || entries_[i].n > entries_[(size_t) best].n)) best = (int) i;
        Job j;
        if (best >= 0 && entries_[(size_t) best].n == n) {   // the same boundary again: only its use time
            entries_[(size_t) best].used = now_s();
            return true;
        }
        Entry e;
        if (best >= 0) {
            e = entries_[(size_t) best];
            entries_.erase(entries_.begin() + best);
            j.old_regions = regions_name(e.n);
        } else {
            static int seq = 0;
            e.dir = dir_ + "/rec-" + std::to_string(now_ms()) + "-" + std::to_string(++seq);
            e.rows.assign(streams_.size(), 0);
        }
        ggml_backend_synchronize(dense.backend());
        j.dir = e.dir;
        for (size_t s = 0; s < streams_.size(); ++s) {
            const auto & st = streams_[s];
            const int64_t r0 = e.rows[s], r1 = std::min<int64_t>(st.rows, n / st.ratio);
            if (r1 <= r0) continue;
            Job::Rows w;
            w.stream = (int) s;
            w.offset = (size_t) r0 * st.row_bytes;
            w.data.resize((size_t) (r1 - r0) * st.row_bytes);
            ggml_backend_tensor_get(st.t, w.data.data(), w.offset, w.data.size());
            j.rows.push_back(std::move(w));
            e.rows[s] = r1;
        }
        j.regions.resize(region_bytes_);
        size_t off = 0;
        for (ggml_tensor * t : regions_) {
            ggml_backend_tensor_get(t, j.regions.data() + off, 0, ggml_nbytes(t));
            off += ggml_nbytes(t);
        }
        e.n = n;
        e.tokens.assign(tokens, tokens + n);
        e.ident.clear();
        bool img = false;
        for (size_t i = 0; i < ident_n && (int64_t) i < n; ++i) img |= ident[i] != 0;
        if (img) { e.ident.assign(ident, ident + std::min<size_t>(ident_n, (size_t) n)); e.ident.resize((size_t) n, 0); }
        e.logits = logits && n_logits;
        if (e.logits) j.logits.assign(logits, logits + n_logits);
        e.used = now_s();
        e.bytes = 0;
        for (size_t s = 0; s < streams_.size(); ++s) e.bytes += (size_t) e.rows[s] * streams_[s].row_bytes;
        e.bytes += region_bytes_ + (size_t) n * 4 + e.ident.size() * 8 + j.logits.size() * 4;
        j.entry = e;
        entries_.push_back(e);
        push(std::move(j));
        evict_locked();
        (void) err;
        return true;
    }

    /// The record that is the longest prefix (token and image key) of the prompt: its index, or -1.
    int best(const int * ids, const uint64_t * ident, size_t ident_n, int64_t n_ids) const {
        std::lock_guard<std::mutex> lk(m_);
        int b = -1;
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].n <= n_ids && prefix(entries_[i], ids, ident, ident_n, entries_[i].n) &&
                (b < 0 || entries_[i].n > entries_[(size_t) b].n)) b = (int) i;
        return b;
    }
    int64_t boundary(int i) const { std::lock_guard<std::mutex> lk(m_); return entries_[(size_t) i].n; }
    bool has_logits(int i) const { std::lock_guard<std::mutex> lk(m_); return entries_[(size_t) i].logits; }

    /// Write record i's state into the device (the stream rows and the snapshot tensors): the caller then does
    /// rearm(n) + restore().  logits: the boundary's last-row logits (empty if none were saved).
    bool restore(Ds41Dense & dense, int i, std::vector<float> & logits, std::string & err) {
        flush();
        Entry e;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (i < 0 || (size_t) i >= entries_.size()) { err = "no such record"; return false; }
            entries_[(size_t) i].used = now_s();
            e = entries_[(size_t) i];
        }
        ggml_backend_synchronize(dense.backend());
        std::vector<uint8_t> buf;
        for (size_t s = 0; s < streams_.size(); ++s) {
            const size_t total = (size_t) e.rows[s] * streams_[s].row_bytes;
            if (!total) continue;
            std::ifstream f(e.dir + "/s" + std::to_string(s) + ".bin", std::ios::binary);
            for (size_t off = 0; off < total;) {
                const size_t len = std::min<size_t>(total - off, (size_t) 64 << 20);
                buf.resize(len);
                if (!f.read((char *) buf.data(), (std::streamsize) len)) { err = "short read in " + e.dir; return false; }
                ggml_backend_tensor_set(streams_[s].t, buf.data(), off, len);
                off += len;
            }
        }
        {
            std::ifstream f(e.dir + "/" + regions_name(e.n), std::ios::binary);
            buf.resize(region_bytes_);
            if (!f.read((char *) buf.data(), (std::streamsize) region_bytes_)) { err = "short read of the rings in " + e.dir; return false; }
            size_t off = 0;
            for (ggml_tensor * t : regions_) { ggml_backend_tensor_set(t, buf.data() + off, 0, ggml_nbytes(t)); off += ggml_nbytes(t); }
        }
        logits.clear();
        if (e.logits) {
            std::ifstream f(e.dir + "/logits.bin", std::ios::binary | std::ios::ate);
            const std::streamsize sz = f.tellg();
            if (sz > 0) {
                logits.resize((size_t) sz / 4);
                f.seekg(0);
                f.read((char *) logits.data(), sz);
            }
        }
        ggml_backend_synchronize(dense.backend());
        return true;
    }

    /// Delete every record (waits for the writes in flight).  -> records removed.
    size_t clear() {
        flush();
        std::lock_guard<std::mutex> lk(m_);
        const size_t n = entries_.size();
        std::error_code ec;
        for (const auto & e : entries_) std::filesystem::remove_all(e.dir, ec);
        entries_.clear();
        return n;
    }

    /// Wait until every queued write is on disk.
    void flush() {
        std::unique_lock<std::mutex> lk(qm_);
        qc_.wait(lk, [&] { return q_.empty() && !busy_; });
    }

private:
    static constexpr uint64_t kVersion = 1;
    struct Entry {
        std::string dir;
        int64_t n = 0;
        std::vector<int> tokens;
        std::vector<uint64_t> ident;   // empty: all text
        std::vector<int64_t> rows;     // rows on disk per stream
        size_t bytes = 0;
        int64_t used = 0;
        bool logits = false;
    };
    struct Job {
        std::string dir, old_regions;
        struct Rows { int stream; size_t offset; std::vector<uint8_t> data; };
        std::vector<Rows> rows;
        std::vector<uint8_t> regions;
        std::vector<float> logits;
        Entry entry;
        bool remove = false;           // a deletion (an evicted record)
    };

    static int64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
    static int64_t now_s() { return now_ms() / 1000; }
    static std::string regions_name(int64_t n) { return "regions-" + std::to_string(n) + ".bin"; }

    static bool prefix(const Entry & e, const int * t, const uint64_t * id, size_t idn, int64_t n) {
        if (e.n > n) return false;
        for (int64_t i = 0; i < e.n; ++i) {
            if (e.tokens[(size_t) i] != t[i]) return false;
            const uint64_t a = (size_t) i < e.ident.size() ? e.ident[(size_t) i] : 0;
            const uint64_t b = id && (size_t) i < idn ? id[i] : 0;
            if (a != b) return false;
        }
        return true;
    }

    bool load_entry(const std::string & dir, Entry & e) const {
        namespace fs = std::filesystem;
        std::ifstream m(dir + "/meta.txt");
        if (!m) return false;
        std::string key, fp, regions;
        uint64_t version = 0;
        int logits = 0;
        while (m >> key) {
            if (key == "version") m >> version;
            else if (key == "fp") m >> fp;
            else if (key == "n") m >> e.n;
            else if (key == "regions") m >> regions;
            else if (key == "logits") m >> logits;
            else if (key == "used") m >> e.used;
            else if (key == "rows") { e.rows.assign(streams_.size(), 0); for (auto & r : e.rows) m >> r; }
            else if (key == "ident") { int v; m >> v; e.ident.resize(v ? 1 : 0); }
        }
        if (version != kVersion || fp != fp_ || e.n < 1 || e.rows.size() != streams_.size() || regions != regions_name(e.n))
            return false;
        std::error_code ec;
        e.dir = dir;
        e.logits = logits != 0;
        e.bytes = 0;
        for (size_t s = 0; s < streams_.size(); ++s) {
            const size_t need = (size_t) e.rows[s] * streams_[s].row_bytes;
            if (need && fs::file_size(dir + "/s" + std::to_string(s) + ".bin", ec) < need) return false;
            e.bytes += need;
        }
        if (fs::file_size(dir + "/" + regions, ec) != region_bytes_ || ec) return false;
        e.bytes += region_bytes_;
        std::ifstream t(dir + "/tokens.bin", std::ios::binary);
        e.tokens.resize((size_t) e.n);
        if (!t.read((char *) e.tokens.data(), (std::streamsize) (e.n * 4))) return false;
        e.bytes += (size_t) e.n * 4;
        if (!e.ident.empty()) {
            std::ifstream f(dir + "/ident.bin", std::ios::binary);
            e.ident.resize((size_t) e.n);
            if (!f.read((char *) e.ident.data(), (std::streamsize) (e.n * 8))) return false;
            e.bytes += (size_t) e.n * 8;
        }
        if (e.logits) e.bytes += (size_t) fs::file_size(dir + "/logits.bin", ec);
        return !ec;
    }

    void write_job(Job & j) {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (j.remove) { fs::remove_all(j.dir, ec); return; }
        fs::create_directories(j.dir, ec);
        for (const auto & w : j.rows) {   // at its offset: a longer file left by an earlier crash is overwritten
            const std::string p = j.dir + "/s" + std::to_string(w.stream) + ".bin";
            std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
            if (!f) f.open(p, std::ios::binary | std::ios::out);
            f.seekp((std::streamoff) w.offset);
            f.write((const char *) w.data.data(), (std::streamsize) w.data.size());
        }
        auto put = [&](const std::string & name, const void * data, size_t n) {
            const std::string tmp = j.dir + "/" + name + ".tmp";
            {
                std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
                f.write((const char *) data, (std::streamsize) n);
            }
            fs::rename(tmp, j.dir + "/" + name, ec);
        };
        const Entry & e = j.entry;
        put(regions_name(e.n), j.regions.data(), j.regions.size());
        put("tokens.bin", e.tokens.data(), e.tokens.size() * 4);
        if (!e.ident.empty()) put("ident.bin", e.ident.data(), e.ident.size() * 8);
        if (e.logits) put("logits.bin", j.logits.data(), j.logits.size() * 4);
        std::ostringstream m;
        m << "version " << kVersion << "\nfp " << fp_ << "\nn " << e.n << "\nregions " << regions_name(e.n)
          << "\nlogits " << (e.logits ? 1 : 0) << "\nident " << (e.ident.empty() ? 0 : 1) << "\nused " << e.used << "\nrows";
        for (int64_t r : e.rows) m << ' ' << r;
        m << '\n';
        const std::string s = m.str();
        put("meta.txt", s.data(), s.size());
        if (!j.old_regions.empty() && j.old_regions != regions_name(e.n)) fs::remove(j.dir + "/" + j.old_regions, ec);
    }

    void push(Job && j) {
        {
            std::lock_guard<std::mutex> lk(qm_);
            q_.push_back(std::move(j));
        }
        qc_.notify_all();
    }
    void run() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(qm_);
                qc_.wait(lk, [&] { return !q_.empty() || quit_; });
                if (q_.empty()) return;
                j = std::move(q_.front());
                q_.pop_front();
                busy_ = true;
            }
            write_job(j);
            {
                std::lock_guard<std::mutex> lk(qm_);
                busy_ = false;
            }
            qc_.notify_all();
        }
    }
    void stop() {
        if (!writer_.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(qm_);
            quit_ = true;
        }
        qc_.notify_all();
        writer_.join();
    }
    void evict() { std::lock_guard<std::mutex> lk(m_); evict_locked(); }
    void evict_locked() {
        size_t total = 0;
        for (const auto & e : entries_) total += e.bytes;
        while (total > budget_ && entries_.size() > 1) {   // the newest record stays even alone over the budget
            auto v = std::min_element(entries_.begin(), entries_.end(),
                                      [](const Entry & a, const Entry & b) { return a.used < b.used; });
            total -= v->bytes;
            Job j;
            j.dir = v->dir;
            j.remove = true;
            entries_.erase(v);
            push(std::move(j));
            ++evictions_;
        }
    }

    bool on_ = false;
    std::string dir_, fp_;
    size_t budget_ = 0, region_bytes_ = 0, evictions_ = 0;
    std::vector<Ds41Dense::StreamTensor> streams_;
    std::vector<ggml_tensor *> regions_;
    mutable std::mutex m_;               // entries_
    std::vector<Entry> entries_;
    std::mutex qm_;                      // the write queue
    std::condition_variable qc_;
    std::deque<Job> q_;
    bool busy_ = false, quit_ = false;
    std::thread writer_;
};

}  // namespace strata::v41
