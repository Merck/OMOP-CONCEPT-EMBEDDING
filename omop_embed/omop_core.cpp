#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <random>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#define SPDLOG_FMT_EXTERNAL
#include <spdlog/spdlog.h>
#define USE_ZSTR 1
#ifdef USE_ZSTR
#include <zstr.hpp>
#endif
#undef NDEBUG
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>     // Numpy bindings
#define CHECK(x) do { if (!(x)) { spdlog::critical("{}:{} {} assertion failed: {}", __FILE__, __LINE__, __FUNCTION__,  #x); abort(); } } while(0)
#include "sync_queue.h"


namespace omop {
    using std::array;
    using std::vector;
    using std::unordered_set;
    using std::unordered_map;
    using std::string;
    using std::ifstream;
    using std::ofstream;
    using spdlog::info;
    using spdlog::warn;
    using spdlog::error;
    using random_engine = std::default_random_engine;

    namespace py = pybind11;

    static constexpr int MAX_SOURCE = 16;

    template <typename S, typename T>
    void write_pod (S &stream, T const *ptr, size_t n = 1) {
        stream.write(reinterpret_cast<char const *>(ptr), sizeof(T) * n);
    }

    template <typename S, typename T>
    void read_pod (S &stream, T *ptr, size_t n = 1) {
        stream.read(reinterpret_cast<char *>(ptr), sizeof(T) * n);
    }

    template <typename T>
    T get (py::dict dict, char const *name, T const &v) {
        if (dict.contains(name)) {
            return dict[name].cast<T>();
        }
        return v;
    }

    template <typename T>
    int randint (int a, int b, T &rng) {
        return std::uniform_int_distribution<int>(a, b)(rng);
    }

    namespace julian {
        // https://stackoverflow.com/questions/13932909/difference-between-two-dates-in-c
        int Julian_A[12] = { 1, 1, 0 };
        int Julian_M[12] = { 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        int day (struct tm *date) {
            int a = Julian_A[date->tm_mon];
            int m = Julian_M[date->tm_mon];
            int y = date->tm_year + 1900 + 4800 - a;
            return date->tm_mday + ((153*m + 2) / 5) + 365*y + y/4 - y/100 + y/400 - 32045;
        }
    }

    int parse_date (char const *p) {
        if (*p == 0) return 0;
        struct tm tm;
        p = strptime(p, "%Y-%m-%d", &tm);
        if (*p != 0) return 0;
        return julian::day(&tm);
    }

    string julian_to_string (int jd) {
        int l, n, i, j, k;
        l = jd+68569;
        n = 4*l/146097;
        l = l-(146097*n+3)/4;
        i = 4000*(l+1)/1461001;
        l = l-1461*i/4+31;
        j = 80*l/2447;
        k = l-2447*j/80;
        l = j/11;
        j = j+2-12*l;
        i = 100*(n-49)+i+l;
        return fmt::format("{:04d}-{:02d}-{:02d}", i, j, k);
    }

    struct Person {
        int64_t id;
        int32_t source = 0; // (source, id) uniquely identifies a person
                            // this assumption is slightly simplified
                            // because same person might appear in different
                            // sources under different IDs.
                            // default source is 0
        int32_t size;

        friend void swap (Person &x, Person &y) {
            std::swap(x.id, y.id);
            std::swap(x.source, y.source);
            std::swap(x.size, y.size);
        }
    };

    struct DateRange {
        // must be very careful
        // at this stage of development,
        // one cannot assume all dates are > 0
        // so using 0 to represent None is dangerous

        int start_date, end_date; // end_date is inclusive
        DateRange (): start_date(0), end_date(-1) {
        }

        DateRange (int b, int e): start_date(b), end_date(e) {
        }

        DateRange (py::object row) {
            start_date = parse_date(row["cohort_start_date"].cast<string>().c_str());
            end_date = parse_date(row["cohort_end_date"].cast<string>().c_str());
        }

        bool overlaps (DateRange r) const {
            CHECK(r.start_date > 0);
            if (r.end_date == 0) r.end_date = r.start_date;
            int b = std::max(start_date, r.start_date);
            int e = std::min(end_date, r.end_date);
            return b <= e;
        }
    };

    static int constexpr DOMAIN_VISIT = 4;

    struct Token {
        // Token can be ordered by (start_date, end_date, domain, concept)
        int32_t domain; // condition = 1
                        // procedure = 2
                        // drug = 3
        int32_t concept;
        int32_t start_date;  // !!!!!!!!!! COULD BE 0 for missing
        int32_t end_date;    // !!!!!!!!!! COULD BE 0 for missing
#ifdef EXTRACT_VISITS
        int64_t visit_occurrence;
#endif

        Token () {
        }

        Token (py::list v) {
            domain = v[0].cast<int32_t>();
            concept = v[1].cast<int32_t>();
            start_date = v[2].cast<int32_t>();
            end_date = v[3].cast<int32_t>();
        }

        py::list py () const {
            py::list one;
            one.append(domain);
            one.append(concept);
            one.append(start_date);
            one.append(end_date);
            return one;
        }

        int duration () const {
            int d = end_date - start_date + 1;
            if (d < 0) d = 0;
            return d;
        }

        int64_t encode_date () const {
            uint64_t s = (uint32_t)start_date;
            uint64_t e = (uint32_t)end_date;
            return (e << uint64_t(32)) | s;
        }

        static DateRange decode_date (int64_t iraw) {
            uint64_t raw = (uint64_t)iraw;
            uint64_t constexpr MASK = 0xFFFFFFFF;
            DateRange dr(uint32_t(raw & MASK), uint32_t((raw >> uint64_t(32)) & MASK));
            return dr;
        }

        static void test_encode () {
            Token token;
            token.start_date = 0xEFFFEFFF;
            token.end_date = 0xDEADBEEF;
            int64_t x = token.encode_date();
            auto dr = decode_date(x);
            if (dr.start_date != token.start_date) {
                error("{:X} != {:X}", dr.start_date, token.start_date);
            }
            if (dr.end_date != token.end_date) {
                error("{:X} != {:X}", dr.end_date, token.end_date);
            }
            CHECK(dr.start_date == token.start_date);
            CHECK(dr.end_date == token.end_date);
        }
    };

    static inline bool equal (Token const &t1, Token const &t2) {
        return (t1.concept == t2.concept) && (t1.start_date == t2.start_date) && (t1.end_date == t2.end_date);
    }

    bool operator < (Token const &t1, Token const &t2) {
        if (t1.start_date == t2.start_date) {
            if (t1.end_date == t2.end_date) {
                // by domain
                if (t1.domain == t2.domain) {
                    return t1.concept < t2.concept;
                }
                return t1.domain < t2.domain;
            }
            return t1.end_date < t2.end_date;
        }
        return t1.start_date < t2.start_date;
    }

    struct Record {
        Person person;
        vector<Token> tokens;
        friend void swap (Record &x, Record &y) {
            std::swap(x.person, y.person);
            x.tokens.swap(y.tokens);
        }
    };

    void dedupe_tokens (vector<Token> *tokens) {
        auto it = std::unique(tokens->begin(), tokens->end(), equal);
        tokens->erase(it, tokens->end());
    }

    class BlackList {
        // TODO: implement
        array<unordered_set<int64_t>, MAX_SOURCE> data;
    public:
        BlackList () {
        }

        void load (string const &path) {
            int src;
            int64_t id;
            ifstream is(path);
            int cc = 0;
            while (is >> src >> id) {
                CHECK((src >= 0) && (src < MAX_SOURCE));
                data[src].insert(id);
                ++cc;
            }
            info("loaded {} black list items from {}.", cc, path);
        }

        bool contains (Person const &p) const {
            return data[p.source].count(p.id) > 0;
        }
    };

    class Filter {
        BlackList black;
        int min_tokens;
    public:
        Filter (): min_tokens(0) {}
        Filter (py::dict conf): min_tokens(get(conf, "min_tokens", 10)) {
            if (conf.contains("black_list")) {
                black.load(conf["black_list"].cast<string>());
            }
        }
        bool keep (Record const &r) const {
            if (r.tokens.size() < min_tokens) return false;
            if (black.contains(r.person)) return false;
            return true;
        }
    };

    // returns filled
    int load_partition (string const &path, vector<Record> *records, Filter const &filter, bool append = false) {
        int32_t sz;
#ifdef USE_ZSTR
        zstr::ifstream is(path);
#else
        ifstream is(path, std::ios::binary);
#endif 
        read_pod(is, &sz);
        if (append) {
            records->reserve(records->size() + sz);
        }
        else {
            records->clear();
            records->reserve(sz);
        }
        int added = 0;
        Record r;
        for (int i = 0; i < sz; ++i) {
            int32_t sz1;
            read_pod(is, &r.person);
            read_pod(is, &sz1);
            r.tokens.resize(sz1);
            read_pod(is, &r.tokens[0], sz1);
            dedupe_tokens(&r.tokens);
            /*
            if (r.person.id ==  1446954) {
                info("YYY {}", filter.keep(r));
                for (auto const &t: r.tokens) {
                    info("yyy {} : {} - {}", t.concept, t.start_date, t.end_date);
                }
            }
            */
            if (filter.keep(r)) {
                records->emplace_back(std::move(r));
                ++added;
            }
        }
        return added;
    }

    class Partition {
    public:
        vector<Record> records;
        //unordered_map<int, unordered_map<uint64_t>, int>> lookup;
        vector<unordered_map<int64_t, int>> lookups;
    public:
        static inline size_t file_size (int persons, int tokens) {
            return sizeof(int32_t) + persons * (sizeof(Person) + sizeof(int32_t)) + tokens * sizeof(Token);
        }

        Partition (string const &path) {
            Filter filter;
            load_partition(path, &records, filter);
            int max_src = 0;
            for (int i = 0; i < records.size(); ++i) {
                int src = records[i].person.source;
                if (src > max_src) max_src = src;
            }
            lookups.resize(max_src + 1);
            for (int i = 0; i < records.size(); ++i) {
                auto const &p = records[i].person;
                lookups[p.source][p.id] = i;
            }
        }

        int size () const { return records.size(); }

        py::tuple get (int i) const {
            auto const &r = records[i];
            py::list l;
            for (auto const &t: r.tokens) {
                py::list one;
                one.append(t.domain);
                one.append(t.concept);
                one.append(t.start_date);
                one.append(t.end_date);
                l.append(one);
            }
            return py::make_tuple(r.person.id, l);
        }

        Record const *get_person_record (int source, int64_t pid) const {
            CHECK(source < lookups.size());
            auto const &lookup = lookups[source];
            auto it = lookup.find(pid);
            if (it == lookup.end()) return nullptr;
            return &records[it->second];
        }

        py::tuple get_person (int source, int64_t pid) const {
            CHECK(source < lookups.size());
            auto const &lookup = lookups[source];
            auto it = lookup.find(pid);
            CHECK(it != lookup.end());
            return get(it->second);
        }

    };

    class Stat {
        vector<int> lengths;
        vector<int> times1;
        int times1_zero;
        vector<int> times2;
        int times2_zero;
        vector<int> times3;
        int times3_zero;
    public:
        Stat () {
            times1_zero = times2_zero = times3_zero = 0;
        }
        void update (Partition const *part) {
            for (auto const &r: part->records) {
                lengths.push_back(r.tokens.size());
                for (int i = 1; i < r.tokens.size(); ++i) {
                    int x = r.tokens[i].start_date - r.tokens[i-1].start_date;
                    if (x == 0) {
                        times1_zero += 1;
                    }
                    else {
                        times1.push_back(x);
                    }
                    x = r.tokens[i].start_date - r.tokens[i-1].end_date;
                    if (x == 0) {
                        times2_zero += 1;
                    }
                    else {
                        times2.push_back(x);
                    }
                }
                for (int i = 0; i < r.tokens.size(); ++i) {
                    int x = r.tokens[i].end_date - r.tokens[i].start_date;
                    if (x == 0) {
                        times3_zero += 1;
                    }
                    else {
                        times3.push_back(x);
                    }
                }
            }
        }

        static void report_one (vector<int> &v, string const &name, int z) {
            std::sort(v.begin(), v.end());
            int N = 10;
            std::cout << fmt::format("{}: {}/{} {:.4f}", name, z, v.size(), 1.0 * z/(z + v.size())) << std::endl;
            if (v.empty()) return;
            for (int i = 0; i <= N; ++i) {
                int off = v.size() * i / N;
                if (i == N) --off;
                std::cout << fmt::format("{} {}/{}: {}", name, i, N, v[off]) << std::endl;
            }
        }

        void report () {
            report_one(lengths, "length", 0);
            report_one(times1, "time1", times1_zero);
            report_one(times2, "time2", times2_zero);
            report_one(times3, "time3", times3_zero);
        }

        py::dict get () {
            py::dict r;
            /*
            r["lengths"] = xt::xtensor<int, 2>({lengths.size()}, &lengths[0]);
            r["times1"] = xt::xtensor<int, 2>({times1.size()}, &times1[0]);
            r["times2"] = xt::xtensor<int, 2>({times2.size()}, &times2[0]);
            r["times3"] = xt::xtensor<int, 2>({times3.size()}, &times3[0]);
            */
            return r;
        }
    };



    class PartitionLoader {
        int source;
        unordered_map<int64_t, Record> all;
        size_t token_cnt;
    public:
        PartitionLoader (int src=0): source(src), token_cnt(0) {
            CHECK((source >= 0) && (source < MAX_SOURCE));
        }

        void load_file (string const &path,
                        int domain,
                        int columns,
                        int person_id_col,
                        int concept_id_col,
                        int start_date_col,
                        int end_date_col
#ifdef EXTRACT_VISITS
                        ,int visit_occurrence_col
#endif
                        ) {
            bool is_gzip = false;
            FILE *stream = fopen(path.c_str(), "r");
            if (!stream) {
                error("Cannot open {}.", path);
                throw 0;
            }
            {
                // check for zip
                uint16_t magic;
                size_t sz = fread(&magic, 1, sizeof(magic), stream);
                if ((sz == sizeof(magic)) && (magic == 0x8B1F)) {
                    info("loading gzip {}", path);
                    fclose(stream);
                    string cmd = fmt::format("zcat {}", path);
                    stream = popen(cmd.c_str(), "r");
                    if (!stream) {
                        error("Cannot popen {}.", cmd);
                        throw 0;
                    }
                    is_gzip = true;
                }
                else {
                    fseek(stream, 0, SEEK_SET);
                    info("loading {}", path);
                }
            }
            char *lineptr = NULL;
            size_t len = 0;
            char const *col_ptrs[columns];
            for (;;) {
                ssize_t r = getdelim(&lineptr, &len, '\n', stream);
                if (r <= 0) break;
                if (lineptr[r-1] == '\n') { // remove trailing \n
                    --r;
                    lineptr[r] = 0;
                }
                {
                    // split into columns
                    char *p = lineptr;
                    int off = 0;
                    for (;;) {
                        char *next = p;
                        while ((*next) && (*next != ',')) ++next;
                        col_ptrs[off] = p;
                        ++off;
                        if (*next == 0) break;
                        *next = 0;
                        p = next + 1;
                    }
                    if (off != columns) {
                        error("incorrect columns");
                        throw 0;
                    }
                }

                int64_t person_id = std::atoll(col_ptrs[person_id_col]);
                Token token;
                token.domain = domain;
                token.concept = std::atol(col_ptrs[concept_id_col]);
                token.start_date = parse_date(col_ptrs[start_date_col]);
                token.end_date = parse_date(col_ptrs[end_date_col]);
#ifdef EXTRACT_VISITS
                token.visit_occurrence = std::atoll(col_ptrs[visit_occurrence_col]);
#endif
                all[person_id].tokens.push_back(token);
                ++token_cnt;
            }
            free(lineptr);
            if (is_gzip) {
                pclose(stream);
            }
            else {
                fclose(stream);
            }
        }

        size_t save (string const &path) {
            size_t file_size = Partition::file_size(all.size(), token_cnt);
            info("Loaded {} people, {} tokens.", all.size(), token_cnt);
            info("file size should be {}.", file_size);
            int32_t sz = all.size();
#ifdef USE_ZSTR
            zstr::ofstream os(path, std::ios_base::out, Z_BEST_SPEED);
#else
            ofstream os(path, std::ios::binary);
#endif
            
            write_pod(os, &sz);
            for (auto &p: all) {
                auto &person = p.second.person;
                auto &tokens = p.second.tokens;
                person.id = p.first;
                person.source = source;
                person.size = tokens.size();
                std::sort(tokens.begin(), tokens.end());
                dedupe_tokens(&tokens);
                // dedupe
                write_pod(os, &person);
                sz = tokens.size();
                write_pod(os, &sz);
                write_pod(os, &tokens[0], sz);
            }
            return file_size;
        }
    };

    int probe_partition (string const &path) {
#ifdef USE_ZSTR
        zstr::ifstream is(path);
#else
        ifstream is(path, std::ios::binary);
#endif 
        int32_t sz;
        read_pod(is, &sz);
        return sz;
    }


    class BaseDict {
    protected:
        int32_t end;    // max concept id + 1
        struct Item {
            int32_t concept;
            int32_t count;
        };
        vector<Item> vocab;
    public:
#if 0   // binary, do not use
        void save (string const &path) {
            ofstream os(path, std::ios::binary);
            write_pod(os, &vocab[0], vocab.size());
        }
        void load (string const &path) {
            ifstream is(path, std::ios::binary);
            is.seekg(0, std::ios::end);
            size_t sz = is.tellg();
            vocab.resize(sz / sizeof(Item));
            is.seekg(0, std::ios::beg);
            read_pod(is, &vocab[0], vocab.size());
            end = 0;
            for (auto const &item: vocab) {
                end = std::max(end, item.concept);
            }
            end += 1;
        }
#endif
        void save (string const &path) {
            ofstream os(path);
            for (auto const &v: vocab) {
                os << v.concept << '\t' << v.count << std::endl;
            }
        }
        void load (string const &path) {
            ifstream is(path);
            Item v;
            end = 0;
            vocab.clear();
            while (is >> v.concept >> v.count) {
                vocab.push_back(v);
                end = std::max(end, v.concept);
            }
            end += 1;
            if (vocab.empty()) {
                error("dictionary file {} is empty", path);
                CHECK(0);
            }
        }
    };

    class DictBuilder: public BaseDict {
    public:
        DictBuilder (vector<string> const &paths) {
            Filter filter;
            unordered_map<int32_t, int32_t> count;
            int32_t max_concept = 0;
            int done = 0;
#pragma omp parallel for
            for (int i = 0; i< paths.size(); ++i) {
                vector<Record> samples;
                load_partition(paths[i], &samples, filter, true);
#pragma omp critical
                {
                    for (auto const &r: samples) {
                        for (auto const &t: r.tokens) {
                            count[t.concept] += 1;
                            max_concept = std::max(max_concept, t.concept);
                        }
                    }
                    ++done;
                    info("{} of {} done.", done, paths.size());
                }
            }
            info("max concept id: {}.", max_concept);
            end = max_concept + 1;
            for (auto p: count) {
                Item item;
                item.concept = p.first;
                item.count = p.second;
                vocab.push_back(item);
            }
            std::sort(vocab.begin(), vocab.end(), [](Item const &a, Item const &b) { return a.count > b.count;});
        }
    };

    void build_dict (vector<string> const &paths, string output) {
        DictBuilder builder(paths);
        builder.save(output);
    }

    class SampleLoader {
        Filter filter;
        bool shuffle;
        bool loop;
        int pool_min;
        int pool_max;
        vector<string> paths;
        vector<Record> pool;
        int next_file;
        int files_loaded = 0;
    public:
        SampleLoader (py::dict conf)
            : filter(conf),
            shuffle(conf["shuffle"].cast<bool>()),
            loop(conf["loop"].cast<bool>()),
            pool_min(get(conf, "pool_min", 1000)),
            pool_max(get(conf, "pool_max", 5000)),
            paths(conf["paths"].cast<vector<string>>()),
            next_file(paths.size()), // shuffle upon first loop
            files_loaded(0)
        {
            CHECK(pool_min >= 0 && pool_min < 100000000);
            CHECK(pool_max >= 0 && pool_max < 100000000);
        }

        bool set_loop (bool b) {
            bool old = loop;
            loop = b;
            return old;
        }

        void shuffle_paths (SampleLoader const &ref, random_engine &rng) {
            //CHECK(paths.size() > 1); //<< "There must be at least 2 partitions.";
            for (;;) {
                std::shuffle(paths.begin(), paths.end(), random_engine(rng()));
                if (paths[0] != ref.paths[0]) break;
                info("retrying shuffle paths");
            }
        }

        int load (vector<Record> *samples, int size, random_engine &rng) {
            int samples_loaded = 0;
            for (; samples_loaded < size; ++samples_loaded) {
                if (pool.size() < pool_min) {
                    int filled = 0;
                    // refill pool by loading partitions
                    while (pool.size() < pool_max) {
                        //LOG(INFO) << "XXX " << samples.size() << " " << batch_size;
                        if (next_file >= paths.size()) {
                            if ((files_loaded > 0) && !loop) break;
                            next_file = 0;
                            if (shuffle) {
                                std::shuffle(paths.begin(), paths.end(), random_engine(rng()));
                            }
                        }
                        string const &path = paths[next_file++];
                        filled += load_partition(path, &pool, filter, true);
                        ++files_loaded;
                    }
                    if (filled > 0 && shuffle) {
                        std::shuffle(pool.begin(), pool.end(), random_engine(rng()));
                    }
                }
                if (pool.empty()) break;
                samples->emplace_back();
                swap(samples->back(), pool.back());
                pool.pop_back();
            }
            return samples_loaded;
        }
    };

    template <typename T>
    class Streamer {
        typedef T Loader;
        typedef typename T::Batch Batch;
        Loader loader;
        int loaded;
        int epoch_size;
    protected:
        SyncQueue<Batch *> queue;
        std::thread thread;
        void loader_thread () {
            for (;;) {
                Batch *batch = loader.next();
                if (!batch) break;
                if (!queue.enqueue(batch)) {
                    delete batch;
                    break;
                }
            }
            queue.enqueue();
        }
    public:
        Streamer (py::dict conf)
            : loader(conf),
            loaded(0),
            queue(get(conf, "prefetch", 512)),
            epoch_size(conf["epoch_size"].cast<int>())
        {
            if (conf.contains("save_whitelist")) {
                CHECK(0);
            }
            thread = std::thread([this](){this->loader_thread();});
        }

        ~Streamer () {
            queue.stop();
            thread.join();
            queue.clear([](Batch *b){delete b;});
        }

        void reset () {
            // do nothing
            // this is because the disk streamer
            // is supposed to loop endlessly
            loaded = 0;
        }

        int __len__ () const {
            return epoch_size;
        }

        Streamer *__iter__ () {
            return this;
        }

        py::dict __next__ () {
            // if ((loaded >= total) && !loop) throw pybind11::stop_iteration();
            if (loaded >= epoch_size) throw pybind11::stop_iteration();
            Batch *batch;
            if (!queue.dequeue(&batch)) {
                throw pybind11::stop_iteration();
            }
            py::dict v = batch->py();
            delete batch;
            ++loaded;
            return v;
        }

        void save_whitelist (string const &path) const {
            loader.save_whitelist(path);
        }

        static void expose (py::handle module, const char *name) {
            py::class_<Streamer<T>>(module, name)
                .def(py::init<py::dict>())
                .def("reset", &Streamer::reset)
                .def("__len__", &Streamer<T>::__len__)
                .def("__next__", &Streamer<T>::__next__)
                .def("__iter__", &Streamer<T>::__iter__)
                .def("save_whitelist", &Streamer<T>::save_whitelist)

                ;
        }
    };

    template <typename T>
    class SmallStreamer {
        typedef T Loader;
        typedef typename T::Batch Batch;
        bool loop;
        Loader loader;
        vector<Batch *> batches;
        int offset;
    public:
        SmallStreamer (py::dict conf)
            : loop(conf["loop"].cast<bool>()),
            loader(conf)
        {
            CHECK(!loop);
            info("warming up");
            reset();
            if (conf.contains("save_whitelist")) {
                loader.save_whitelist(conf["save_whitelist"].cast<string>());
            }
        }

        ~SmallStreamer () {
        }

        void reset () {
            for (Batch *b: batches) {
                delete b;
            }
            batches.clear();
            loader.reset();
            for (;;) {
                Batch *b = loader.next();
                if (!b) break;
                batches.push_back(b);
            }
            //info("loaded {} batches.", batches.size());
            offset = 0;
        }

        int __len__ () const {
            return batches.size();
        }

        SmallStreamer *__iter__ () {
            CHECK(0);
            reset();
            return this;
        }

        py::dict __next__ () {
            if (offset >= batches.size()) {
                throw pybind11::stop_iteration();
            }
            Batch *batch = batches[offset++];
            py::dict v = batch->py();
            return v;
        }

        static void expose (py::handle module, const char *name) {
            py::class_<SmallStreamer<T>>(module, name)
                .def(py::init<py::dict>())
                .def("reset", &SmallStreamer::reset)
                .def("__len__", &SmallStreamer<T>::__len__)
                .def("__next__", &SmallStreamer<T>::__next__)
                .def("__iter__", &SmallStreamer<T>::__iter__)
                ;
        }
    };

    template <typename T>
    class MemoryStreamer {
        typedef T Loader;
        typedef typename T::Batch Batch;
        bool loop;
        int offset = 0;
        vector<Batch *> batches;
    public:
        MemoryStreamer (py::dict conf)
            : loop(conf["loop"].cast<bool>())
        {
            py::dict v;
            v.attr("update")(conf);
            v["loop"] = false;
            Loader loader(v);
            for (;;) {
                Batch *b = loader.next();
                if (!b) break;
                batches.push_back(b);
            }
            if (conf.contains("save_whitelist")) {
                loader.save_whitelist(conf["save_whitelist"].cast<string>());
            }
            info("{} batches loaded.", batches.size());
        }

        ~MemoryStreamer () {
            for (Batch *b: batches) delete b;
        }

        void reset () {
            offset = 0;
        }

        int __len__ () const {
            return batches.size();
        }

        MemoryStreamer *__iter__ () {
            reset();
            return this;
        }

        py::dict __next__ () {
            if (offset >= batches.size()) {
                if (!loop) throw pybind11::stop_iteration();
                offset = 0;
            }
            Batch *batch = batches[offset++];
            py::dict v = batch->py();
            return v;
        }

        static void expose (py::handle module, const char *name) {
            py::class_<MemoryStreamer<T>>(module, name)
                .def(py::init<py::dict>())
                .def("reset", &MemoryStreamer::reset)
                .def("__len__", &MemoryStreamer<T>::__len__)
                .def("__next__", &MemoryStreamer<T>::__next__)
                .def("__iter__", &MemoryStreamer<T>::__iter__)
                ;
        }
    };

    namespace loaders {

        class BertDict: BaseDict {
            int vocab_size;         // total size of embedding vocabulary,
                                    // including special tokens
            vector<int> vocab_lookup;
                                    // map original concept to embedding vocab
            mutable vector<int> accum;  // count of words looked-up
            std::uniform_int_distribution<int> uniform;
        public:
            enum {
                TOKEN_PAD = 0,
                TOKEN_UNK,
                TOKEN_EOS,
                TOKEN_SOS,
                TOKEN_MASK,
                SPECIAL_TOKENS
            };

            BertDict (py::dict conf)
                : vocab_size(conf["vocab_size"].cast<int>()),
                uniform(SPECIAL_TOKENS, vocab_size-1)
            {
                CHECK(vocab_size > SPECIAL_TOKENS);
                load(conf["vocab"].cast<string>());
                vocab_lookup.resize(end, TOKEN_UNK);
                int n = vocab_size - SPECIAL_TOKENS;
                CHECK(n >= 0);
                vocab.resize(n);
                for (int i = 0; i < n; ++i) {
                    int c = vocab[i].concept;
                    CHECK(c >= 0 && c < end);
                    vocab_lookup[c] = SPECIAL_TOKENS + i;
                }
                accum.resize(end, 0);
                info("dict loaded end: {}, size: {}", end, vocab_size);
                if (conf.contains("whitelist")) {
                    string wl = conf["whitelist"].cast<string>();
                    int th = get<int>(conf, "whitelist_threshold", 1);
                    apply_whitelist(wl, th);
                }
            }

            void apply_whitelist (string const &path, int th) {
                ifstream is(path);
                int c, v;
                vector<int> cnt(end, 0);
                while (is >> c >> v) {
                    cnt[c] = v;
                }
                int cc = 0;
                int kept = 0;
                for (int c = 0; c < end; ++c) {
                    if (cnt[c] < th) {
                        ++cc;
                        vocab_lookup[c] = TOKEN_UNK;
                    }
                    else {
                        ++kept;
                    }
                }
                info("applied whitelist {}, masking out {} concepts.", path, cc);
                info("keeping {} concepts", kept);
                CHECK(kept);
            }

            void save_accum (string const &path) const {
                info("saving whitelist to {}", path);
                ofstream os(path);
                int cc = 0;
                for (int i = 0; i < end; ++i) {
                    if (accum[i] > 0) {
                        os << i << '\t' << accum[i] << std::endl;
                        ++cc;
                    }
                }
                info("{} entries saved", cc);
                CHECK(cc);
            }

            size_t size () const {
                return vocab_size;
            }

            py::list dict () const {
                // vocab_lookup[concept] = token | TOKEN_UNK
                vector<int> r(vocab_size, -1);
                for (int i = 0; i < vocab_lookup.size(); ++i) {
                    int token = vocab_lookup[i];
                    // token < vocab_size
                    if (token != TOKEN_UNK) {
                        r[token] = i;
                    }
                }
                for (int i = 0; i < SPECIAL_TOKENS; ++i) {
                    CHECK(r[i] == -1);
                }
                /*
                for (int i = SPECIAL_TOKENS; i < r.size(); ++i) {
                    if (!(r[i] >= 0)) {
                        error("{}: {}", i, r[i]);
                    }
                }
                */
                py::list l;
                for (int v: r) {
                    l.append(v);
                }
                return l;
            }

            py::list special () const {
                py::list r;
                r.append("pad");
                r.append("unk");
                r.append("eos");
                r.append("sos");
                r.append("mask");
                return r;
            }

            int lookup (int c) const {
                if ((c >= 0) && (c < end)) {
                    ++accum[c];
                    return vocab_lookup[c];
                }
                return TOKEN_UNK;
            }

            int random (random_engine &e) {
                return uniform(e);
            }
        };

        class Base {
        protected:
            random_engine rng;
            SampleLoader loader;
            int batch_size;
            int seq_len;
            BertDict dict;
            Base (py::dict conf):
                rng(get(conf, "seed", 2021)),
                loader(conf),
                batch_size(get(conf, "batch_size", 16)),
                seq_len(get(conf, "seq_len", 512)),
                dict(conf)
            {
                // build dictionary
                CHECK(seq_len > 3);
            }

        public:
            void reset () {
                CHECK(0);
            }

            void save_whitelist (string const &path) const {
                dict.save_accum(path);
            }
        };

        class ControlPool {
            vector<int> pool;
            int vocab_size;
        public:
            ControlPool (int vocab_size_): vocab_size(vocab_size_) {
                CHECK(vocab_size > 0);
            }

            int get (random_engine &rng) {
                if (pool.empty()) {
                    for (int i = 0; i < vocab_size; ++i) {
                        pool.push_back(i);
                    }
                    std::shuffle(pool.begin(), pool.end(), random_engine(rng()));
                }
                int v = pool.back();
                pool.pop_back();
                return v;
            }

            int exchange (int v, random_engine &rng) {
                if (pool.empty()) {
                    for (int i = 0; i < vocab_size; ++i) {
                        pool.push_back(i);
                    }
                    std::shuffle(pool.begin(), pool.end(), random_engine(rng()));
                }
                int i = rng() % pool.size();
                int x = pool[i];
                pool[i] = v;
                return x;
            }
        };

        class Basic: public Base {
        protected:
            struct Meta {
                static int64_t constexpr source = 0;
                int64_t pid;
                int64_t label;
                int64_t index;
                int64_t cutoff;
                Meta *next;
                Record const *record;
            };
            enum {
                SPLIT_TRAIN = 1,
                SPLIT_TEST = 2,
                SPLIT_BOTH = 3
            };
            bool is_inference;
            int split_selection;
            bool has_meta;
            double drop_rate;
            vector<Meta> metas;
            // refers &metas[?], do not call delete
            // Meta * is chained by next
            // if load_all is true, meta_lookup is cleared after initial setup
            unordered_map<int64_t, Meta *> meta_lookup;

            bool local;
            int local_offset;
            bool local_shuffle;
            bool local_loop;
            vector<Record> local_records;
            Record dummy_record;

            int nz_cnt = 0;
            int reset_cnt = 0;
            int cutoff_cnt = 0;
            int total_cnt = 0;
            int loaded_batches = 0;
            bool pretrain = false;
            bool no_control_pool = false;
            ControlPool control_pool;
        public:
            struct Batch {
                unsigned batch_size, seq_len, pretrain;
                xt::xtensor<int64_t, 2> raw;
                xt::xtensor<int64_t, 2> time;
                xt::xtensor<int64_t, 2> duration;
                xt::xtensor<int64_t, 2> tokens;
                xt::xtensor<int64_t, 2> mask;
                xt::xtensor<int64_t, 1> labels;
                xt::xtensor<int64_t, 1> patients;
                xt::xtensor<int64_t, 1> sources;
                //xt::xtensor<int64_t, 1> lengths;
                xt::xtensor<int64_t, 2> pretrain_mask;
                xt::xtensor<int64_t, 2> pretrain_labels;
                xt::xtensor<int64_t, 2> pretrain_control;
                xt::xtensor<int64_t, 2> pretrain_gaps;

                //vector<Record> records;
                py::dict py () const {
                    py::dict v;
                    v["raw"] = raw;
                    v["time"] = time;
                    v["duration"] = duration;
                    v["tokens"] = tokens;
                    v["mask"] = mask;
                    v["labels"] = labels;
                    v["patients"] = patients;
                    v["sources"] = sources;
                    //v["lengths"] = lengths;
                    if (pretrain) {
                        v["pretrain_labels"] = pretrain_labels;
                        v["pretrain_control"] = pretrain_control;
                        v["pretrain_gaps"] = pretrain_gaps;
                        v["pretrain_mask"] = pretrain_mask;
                    }
                    return v;
                }

                Batch (unsigned B, unsigned L, bool pretrain_)
                    : batch_size(B), seq_len(L), pretrain(pretrain_)
                {
                    raw.resize({B, L});
                    time.resize({B, L});
                    duration.resize({B, L});
                    tokens.resize({B, L});
                    mask.resize({B, L});
                    labels.resize({B});
                    patients.resize({B});
                    sources.resize({B});
                    //lengths.resize({B});
                    if (pretrain) {
                        pretrain_labels.resize({B, L});
                        pretrain_control.resize({B, L});
                        pretrain_gaps.resize({B, L});
                        pretrain_mask.resize({B, L});
                    }
                }
            };

            Basic (py::dict conf)
                : Base(conf),
                //rand1(0, 1),
                //max_sent(seq_len - 2),
                is_inference(get<bool>(conf, "inference", false)),
                split_selection(get<int>(conf, "split", SPLIT_BOTH)),
                has_meta(false),
                drop_rate(get<double>(conf, "drop_rate", 0)),
                local(get<bool>(conf, "local", false)),
                local_offset(0),
                local_shuffle(conf["shuffle"].cast<bool>()),
                local_loop(conf["loop"].cast<bool>()),
                pretrain(get<bool>(conf, "pretrain", false)),
                no_control_pool(get<bool>(conf, "no_control_pool", false)),
                control_pool(dict.size())
                  
            {
                info("basic streamer, is inference: {}", is_inference);
                if (conf.contains("meta")) {
                    loadMeta(conf);
                }
                if (drop_rate > 0) {
                    CHECK(!is_inference);
                    info("using drop_rate {}", drop_rate);
                }
                info("Control Pool: {}", !no_control_pool);
                if (local) {
                    CHECK(has_meta);
                    bool loop_save = loader.set_loop(false);
                    for (;;) {
                        vector<Record> batch;
                        loader.load(&batch, batch_size, rng);
                        if (batch.empty()) break;
                        for (auto const &row: batch) {
                            int64_t pid = row.person.id;
                            if (meta_lookup.find(pid) == meta_lookup.end()) continue;
                            local_records.push_back(row);
                        }
                    }
                    loader.set_loop(loop_save);
                    info("loading {} records into {} metas.", local_records.size(), metas.size());
                    int cnt = 0;
                    for (auto const &row: local_records) {
                        int64_t pid = row.person.id;
                        auto it = meta_lookup.find(pid);
                        CHECK(it != meta_lookup.end());
                        Meta *meta = it->second;
                        while (meta) {
                            CHECK(!meta->record);
                            meta->record = &row;
                            meta = meta->next;
                            ++cnt;
                        }
                    }
                    info("{} records loaded into {} metas.", local_records.size(), metas.size());
                    if (cnt != metas.size()) {
                        warn("{} metas do not have records", metas.size() - cnt);
                    }
                    for (auto &meta: metas) {
                        if (!meta.record) {
                            meta.record = &dummy_record;
                            ++cnt;
                        }
                    }
                    CHECK(cnt == metas.size());
                    meta_lookup.clear();
                    if (local_shuffle) {
                        std::shuffle(metas.begin(), metas.end(), random_engine(rng()));
                    }
                }
                dummy_record.person.id = -1;
                dummy_record.person.source = 0;
                dummy_record.person.size = 0;
            }

            void reset () {
                if (local) {
                    local_offset = 0;
                }
                else {
                }
                if (total_cnt == 0) total_cnt = 1;
                if (reset_cnt < 2) {
                    info("batches: {}, reset nz:{} cutoff:{} total:{}", loaded_batches, nz_cnt, cutoff_cnt, total_cnt);
                    info("nz_r:{:.2f} cutoff_r:{:.2f}", 1.0 * nz_cnt / total_cnt, 1.0 * cutoff_cnt / total_cnt);
                    nz_cnt = 0;
                }
                loaded_batches = 0;
                cutoff_cnt = 0;
                total_cnt = 0;
                ++reset_cnt;

            }

            void loadMeta (py::dict conf) {
                py::object pop = conf["meta"];
                bool do_train = (split_selection & SPLIT_TRAIN) > 0;
                bool do_test = (split_selection & SPLIT_TEST) > 0;
                info("split train: {}, test: {}", do_train, do_test);
                unordered_map<int, int> label_cnts;
                for (auto h: pop.attr("iterrows")()) {
                    Meta meta;
                    auto row = h.cast<py::tuple>()[1];
                    string date = row["cohortStartDate"].cast<string>();
                    meta.pid = row["subjectId"].cast<double>();
                    meta.label = row["outcomeCount"].cast<double>();
                    meta.index = row["indexes"].cast<double>();
                    meta.cutoff = parse_date(date.c_str());
                    meta.next = nullptr;
                    meta.record = nullptr;
                    if ((do_train && (meta.index >= 0)) ||
                        (do_test  && (meta.index < 0))) {
                        metas.push_back(meta);
                        label_cnts[meta.label] += 1;
                    }
                }

                if (conf.contains("oversample_class")) {
                    CHECK(label_cnts.size() == 2);
                    int oversample_class = get<int>(conf, "oversample_class", -1);
                    int oversample_factor = get<int>(conf, "oversample_factor", 0);
                    if (oversample_class == -1) {
                        if (label_cnts[0] < label_cnts[1]) {
                            oversample_class = 0;
                        }
                        else {
                            oversample_class = 1;
                        }
                    }
                    vector<Meta> over;
                    for (auto const &meta: metas) {
                        if (meta.label == oversample_class) {
                            over.push_back(meta);
                        }
                    }
                    if (oversample_factor < 0) {
                        int n = label_cnts[1-oversample_class] - label_cnts[oversample_class];
                        for (int i = 0; i < n; ++i) {
                            metas.push_back(over[i % over.size()]);
                        }
                    }
                    else for (auto const &meta: over) {
                        for (int i = 0; i < oversample_factor; ++i) {
                            metas.push_back(meta);
                        }
                    }
                }

                meta_lookup.clear();
                for (auto &meta: metas) {
                    auto p = meta_lookup.insert(std::make_pair(meta.pid, &meta));
                    if (!p.second) {
                        meta.next = p.first->second;
                        p.first->second = &meta;
                        CHECK(meta.next->pid == meta.pid);
                    }
                }
                has_meta = true;
                info("{} meta samples loaded so far.", metas.size());
                info("corresponding to {} people.", meta_lookup.size());
                for (auto const &p: label_cnts) {
                    info("Label {}: {}", p.first, p.second);
                }
            }

            void local_load (vector<Meta> *samples, int size, random_engine &rng) {
                for (int i = 0; i < size; ++i) {
                    if (local_offset >= metas.size()) {
                        if (!local_loop) return;
                        local_offset = 0;
                        CHECK(local_offset < metas.size());
                        if (local_shuffle) {
                            std::shuffle(metas.begin(), metas.end(), random_engine(rng()));
                        }
                    }
                    samples->push_back(metas[local_offset]);
                    ++local_offset;
                }
            }

            Batch *next () {
                return next_impl();
            }

            Batch *next_impl () { // just so it's easier to locate
                // there are multple methods named next ()
                vector<Record> cache;
                vector<Meta> samples;
                if (local) {
                    local_load(&samples, batch_size, rng);
                }
                else {
                    CHECK(split_selection == SPLIT_BOTH);
                    loader.load(&cache, batch_size, rng);
                    if (has_meta) {
                        for (auto const &row: cache) {
                            auto it = meta_lookup.find(row.person.id);
                            CHECK(it != meta_lookup.end());
                            Meta meta = *it->second;
                            meta.record = &row;
                            samples.push_back(meta);
                        }
                    }
                    else {
                        for (auto const &row: cache) {
                            // generate dummy meta
                            Meta meta;
                            meta.pid = row.person.id;
                            meta.label = 0;
                            meta.index = 0;
                            meta.cutoff = -1;
                            meta.next = nullptr;
                            meta.record = &row;
                            samples.push_back(meta);
                        }
                    }
                }
                if (samples.empty()) return nullptr;
                Batch *batch = new Batch(samples.size(), seq_len, pretrain);

                batch->raw.fill(-1);
                batch->time.fill(0);
                batch->duration.fill(0);
                batch->tokens.fill(BertDict::TOKEN_PAD);
                batch->mask.fill(0);
                if (pretrain) {
                    batch->pretrain_mask.fill(0);
                    batch->pretrain_labels.fill(0);
                    batch->pretrain_control.fill(0);
                    batch->pretrain_gaps.fill(0);
                }

                int off = 0;
                for (Meta const &meta: samples) {
                    auto const &row = *meta.record;
                    batch->labels(off) = meta.label;
                    batch->patients(off) = meta.pid;
                    batch->sources(off) = meta.source;
                    CHECK(meta.source == row.person.source);

                    //TODO: filter tokens to be included
                    vector<unsigned> pick;
                    pick.reserve(row.tokens.size());
                    for (unsigned i = 0; i < row.tokens.size(); ++i) {
                        // check cutoff
                        auto const &token = row.tokens[i];
                        //int xxx = dict.lookup(token.concept);
                        //info("start {}, concept {}, lookup {}", token.start_date, token.concept, xxx);
                        if (token.start_date <= 0) continue;    // TODO warn
                        if (dict.lookup(token.concept) == BertDict::TOKEN_UNK) continue;
                        if ((meta.cutoff > 0) && (token.start_date > meta.cutoff)) continue;
                        pick.push_back(i);
                    }
                    int keep = pick.size();   // sample tokens, how many to keep
                    if ((!is_inference) && (drop_rate > 0)) {
                        int drop = std::floor(pick.size() * drop_rate);
                        keep -= drop;
                    }
                    int seq_len_internal = seq_len - 1;

                    if ((!is_inference) && (keep > seq_len_internal)) {
                        keep = seq_len_internal;
                    }
                    if (keep < pick.size()) {
                        std::shuffle(pick.begin(), pick.end(), random_engine(rng()));
                        pick.resize(keep);
                        std::sort(pick.begin(), pick.end());
                    }

                    //batch->lengths(off) = pick.size();


                    // determine
                    // at this time, tokens might still be longer than seq_len_internal
                    int from = 0;
                    int to = 0;
                    int n = std::min<int>(seq_len_internal, pick.size());
                    if (n < seq_len_internal) {
                        // input smaller than seq_len
                        // right-shift
                        to = seq_len_internal -  n;
                    }
                    else {
                        from = pick.size() - n;
                        if (!is_inference) {
                            // we can randomly shift
                            ;
                        }
                    }

                    total_cnt += seq_len;
                    nz_cnt += n;
                    cutoff_cnt += pick.size() - n;

                    auto mask = xt::view(batch->mask, off, xt::all());
                    auto raw = xt::view(batch->raw, off, xt::all());
                    auto time = xt::view(batch->time, off, xt::all());
                    auto duration = xt::view(batch->duration, off, xt::all());
                    auto tokens = xt::view(batch->tokens, off, xt::all());


                    //std::fill(mask + to, mask_end, 1);

                    // copy  [from, from + n) 
                    // to    [to,   to + n)
                    int from_end = from + n;

                    for (int i = 0; i < n; ++i, ++from, ++to) {
                        auto const &inp = row.tokens[pick[from]];
                        mask(to) = 1;
                        raw(to) = inp.concept;
                        time(to) = inp.start_date - meta.cutoff;
                        // TODO: censoring needed here
                        int end_date = inp.end_date;
                        // HANDLING OF END_DATE, NOTE !!!
                        if (end_date > meta.cutoff) {
                            end_date = meta.cutoff;
                        }
                        int d = end_date - inp.start_date;
                        if (d < 0) d = 0;
                        duration(to) = d;
                        tokens(to) = dict.lookup(inp.concept);

                        if (pretrain) {
    auto pretrain_mask = xt::view(batch->pretrain_mask, off, xt::all());
    auto pretrain_labels = xt::view(batch->pretrain_labels, off, xt::all());
    auto pretrain_control = xt::view(batch->pretrain_control, off, xt::all());
    auto pretrain_gaps = xt::view(batch->pretrain_gaps, off, xt::all());
                            // determine first token with next date
                            //
                            // TODO: this is not optimal but changing will
                            // probably be too complicated
                            //
                            // When we generate pretrain target, we
                            // don't necessarily need to sample from picked,
                            // but we can sample from original sequence
                            // 
                            // but if we sample from original data, we need
                            // to test whether token is UNK again
                            int next = from + 1;
                            for (;;) {
                                if (next >= from_end) break;
                                auto const &next_token = row.tokens[pick[next]];
                                if (next_token.start_date > inp.start_date) break;
                                ++next;
                            }
                            if (next < from_end) {
                                // generate pretrain target
                                int x = rng() % (from_end - next);
                                CHECK(x >= 0);
                                next += x;
                                auto const &next_token = row.tokens[pick[next]];
                                pretrain_mask(to) = 1;
                                pretrain_labels(to) = dict.lookup(next_token.concept);
                                pretrain_gaps(to) = inp.start_date - next_token.start_date;
                                // generate control
                                //
                                if (no_control_pool) {
                                pretrain_control(to) = rng() % dict.size();
                                }
                                else {
                                pretrain_control(to) = control_pool.exchange(tokens[to], rng);
                                }
                            }
                        }
                    }
                    CHECK(seq_len_internal == to);
                    tokens(seq_len_internal) = BertDict::TOKEN_EOS;

                    ++off;
                }
                ++loaded_batches;
                return batch;
            }
        };
    }

#if 0
    // Imlementation of PLP's feature extractor for 
    // end-to-end debug
    class Encoder {
    protected:
        unordered_map<int64_t, DateRange> cohort;
        vector<int> terms;
        int source;
        int vocab_size;
        int seq_len;
        bool by_freq;
        string dict_path;
        int max_cnt;
    private:
        unordered_map<int64_t, int> cnts;
        unordered_map<int64_t, int> lookup;
    public:
        Encoder (py::dict conf)
        {
            py::list pyterms = conf["terms"].cast<py::list>();
            for (auto x: pyterms) {
                terms.push_back(x.cast<int>());
            }
            source = get(conf, "source", 0);
            vocab_size = get(conf, "vocab_size", 0);
            seq_len = get(conf, "seq_len", 0);
            by_freq = get(conf, "by_freq", false);
            info("Encoder using seq_len {} by_freq {}", seq_len, by_freq);
            if (conf.contains("dict")) {
                dict_path = conf["dict"].cast<string>();
                info("Using dict {}", dict_path);
            }
            max_cnt = 0;
            if (by_freq) error("by_freq is not working, do not use.");
            CHECK(!by_freq);
        }

        void updateCohort (py::object df) {
            cohort.clear();
            for (auto handle: df.attr("iterrows")()) {
                py::object row = handle.cast<py::tuple>()[1];
                cohort[row["subject_id"].cast<int64_t>()] = DateRange(row);
            }
        }

        void trainBatch (py::object batch) {
            py::list records = batch["records"].cast<py::list>();
            for (auto h: records) {
                for (auto h: h.cast<py::list>()) {
                    Token token(h.cast<py::list>());
                    if (token.concept > 0) {
                        // 0 is "no matching concept", R does not use it
                        cnts[token.concept] += 1;
                    }
                }
            }
            /*
            xt::xtensor<int64_t, 2> raw = batch["raw"].cast<xt::xtensor<int64_t, 2>>();
            int64_t const *token = raw.data();
            size_t sz = raw.size();
            size_t total = 0;
            for (size_t i = 0; i < sz; ++i) {
                int64_t x = token[i];
                if (x >= 0) {
                    cnts[x] += 1;
                    ++total;
                }
            }
            */
            // info("train extractor batch, total tokens {}", total);
        }

        py::list finishTrain () {
            vector<std::pair<int, int64_t>> xx;
            if (dict_path.size()) {
                ifstream is(dict_path);
                int token, count;
                while (is >> token >> count) {
                    xx.emplace_back(count, token);
                }
                if (xx.empty()) {
                    error("dictionary file {} is empty", dict_path);
                    CHECK(0);
                }
            }
            else {
                for (auto const &p: cnts) {
                    xx.emplace_back(p.second, p.first);
                }
                std::sort(xx.begin(), xx.end());
                std::reverse(xx.begin(), xx.end());
            }
            max_cnt = xx[0].first;
            if ((vocab_size > 0) && (xx.size() > vocab_size)) {
                xx.resize(vocab_size);
            }
            lookup.clear();
            py::list codebook;
            for (int i = 0; i < xx.size(); ++i) {
                lookup[xx[i].second] = i;
                codebook.append(xx[i].second);
            }
            return codebook;
        }

        void encode_one (double *out, int cutoff, int len, int64_t *raw, int64_t *raw_time) const {
            for (int t: terms) {
                DateRange range(cutoff + t, cutoff - 1);
                for (int i = 0; i < len; ++i) {
                    if (raw[i] < 0) continue;
                    auto span = Token::decode_date(raw_time[i]);
                    if (span.start_date <= 0) {
                        error("BAD RECORD {}: {} - {}", raw[i], span.start_date, span.end_date);
                    }
                    if (!range.overlaps(span)) continue;
                    int64_t c = raw[i];
                    auto it = lookup.find(c);
                    if (it == lookup.end()) continue;
                    out[it->second] = 1;
                }
                out += lookup.size();
            }
        }

        xt::xtensor<double, 2> extractPartition (py::object prediction, string const &partition) {
            Partition part(partition);

            if (dict_path.size()) {
                /*
                info("Using dictionary {}", dict_path);
                ifstream is(dict_path);
                int token, count;
                while (is >> token >> count) {
                    cnts[token] = count;
                }
                */
                ;
            }
            else {
                info("Training dictionary");
                // THIS MIGHT LEAK TEST INFO
                for (auto h: part.records) {
                    for (auto const &token: h.tokens) {
                        if (token.concept > 0) {
                            // 0 is "no matching concept", R does not use it
                            cnts[token.concept] += 1;
                        }
                    }
                }
            }
            finishTrain();

            int rows = py::len(prediction);
            info("feature size {} x {} * {}", rows, lookup.size(), terms.size());
            xt::xtensor<double, 2> fts({rows, lookup.size() * terms.size()});
            int off = 0;
            for (auto h: prediction.attr("iterrows")()) {
                auto row = h.cast<py::tuple>()[1];
                int64_t sid = row["subjectId"].cast<double>();
                int32_t start_date = row["cohortStartDate"].cast<double>();
                double *out = &fts(off++);
                Record const *rec = part.get_person_record(source, sid);
                if (rec == nullptr) {
                    warn("Cannot find record for {}: {}", source, sid);
                }
                else {
                    vector<Token> tokens;
                    int n = rec->tokens.size();
                    if (by_freq) {
                        if (n <= seq_len) {
                            tokens = rec->tokens;
                        }
                        else {
                            vector<std::pair<double, double>> index;
                            for (int i = 0; i < rec->tokens.size(); ++i) {
                                auto it = cnts.find(rec->tokens[i].concept);
                                int cnt = 0;
                                if (it != cnts.end()) {
                                    cnt = it->second;
                                }
                                double entropy = 0;
                                if (cnt <= 0 || cnt >= max_cnt) {
                                    ;
                                }
                                else {
                                    double p = 1.0 * cnt / max_cnt;
                                    double q = 1.0 - p;
                                    entropy = - p * std::log(p) - q * std::log(q);
                                }
                                index.emplace_back(entropy, i);
                            }
                            std::sort(index.begin(), index.end());
                            std::reverse(index.begin(), index.end());
                            CHECK(index.size() > seq_len);
                            index.resize(seq_len);
                            for (auto &p: index) {
                                std::swap(p.first, p.second);
                            }
                            std::sort(index.begin(), index.end());
                            for (auto const &p: index) {
                                tokens.push_back(rec->tokens[int(p.first)]);
                            }
                        }
                    }
                    else {
                        int off = 0;
                        if ((seq_len > 0) && (n > seq_len)) {
                            off = n - seq_len;
                            n = seq_len;
                        }
                        tokens = vector<Token>(rec->tokens.begin() + off, rec->tokens.end());
                    }
                    vector<int64_t> raw;
                    vector<int64_t> raw_time;
                    for (auto const &token: tokens) {
                        raw.push_back(token.concept);
                        raw_time.push_back(token.encode_date());
                    }
                    encode_one(out, start_date, raw.size(), &raw[0], &raw_time[0]);
                }
            }
            return fts;
        }

        xt::xtensor<double, 2> extractBatch (py::object batch) const {
            xt::xtensor<int64_t, 2> sid = batch["patients"].cast<xt::xtensor<int64_t, 2>>();
            auto sid_shape = sid.shape();
            int n = sid_shape[0];
            xt::xtensor<double, 2> fts({n, lookup.size() * terms.size()});
            py::list records = batch["records"].cast<py::list>();
#if 1
            int64_t const *sid_ptr = sid.data();
            CHECK(py::len(records) == n);
            for (int i = 0; i < n; ++i) {
                py::list one = records[i].cast<py::list>();
                double *out = &fts(i);
                vector<int64_t> raw;
                vector<int64_t> raw_time;
                for (auto h: one) {
                    Token token(h.cast<py::list>());
                    raw.push_back(token.concept);
                    raw_time.push_back(token.encode_date());
                }
                auto it = cohort.find(sid_ptr[i]);
                CHECK(it != cohort.end());
                if (sid_ptr[i] == 1446954) {
                    info("start date: {}", it->second.start_date);
                }
                encode_one(out, it->second.start_date, raw.size(), &raw[0], &raw_time[0]);
            }
#else
            xt::xtensor<int64_t, 2> raw = batch["raw"].cast<xt::xtensor<int64_t, 2>>();
            xt::xtensor<int64_t, 2> raw_time = batch["raw_time"].cast<xt::xtensor<int64_t, 2>>();
            auto raw_shape = raw.shape();
            auto raw_time_shape = raw.shape();
            CHECK(sid_shape[1] = 1);
            CHECK(n == raw_shape[0]);
            CHECK(n == raw_time_shape[0]);
            int seq_len = raw_shape[1];
            CHECK(seq_len == raw_time_shape[1]);
            int64_t const *sid_ptr = sid.data();
            for (int i = 0; i < sid_shape[0]; ++i) {
                int64_t *raw_ptr = &raw(i);
                int64_t *raw_time_ptr = &raw_time(i);
                double *out = &fts(i);
                auto it = cohort.find(sid_ptr[i]);
                CHECK(it != cohort.end());
                if (sid_ptr[i] == 1446954) {
                    for (int j = 0; j < seq_len; ++j) {
                        auto tr = Token::decode_date(raw_time_ptr[j]);
                        info("XXX {} : {} - {}", raw[j], tr.start_date, tr.end_date);

                    }

                }
                encode_one(out, it->second.start_date, seq_len, raw_ptr, raw_time_ptr);
            }
#endif
            return fts;
        }
    };

    class Embedder: public Encoder {
        loaders::BertDict dict;
        Matrix<float> weights;
        int hidden;

    public:
        Embedder (py::dict conf, Matrix<float> w):
            Encoder(conf),
            dict(conf), weights(w)
        {
            vector<size_t> shape = weights.shape();
            CHECK(shape.size() == 2);
            CHECK(shape[0] == dict.size());
            hidden = shape[1];
        }

        void embed_one (double *out, int cutoff, int len, int64_t *raw, int64_t *raw_time) const {
            for (int t: terms) {
                int cnt = 0;
                DateRange range(cutoff + t, cutoff - 1);
                for (int i = 0; i < len; ++i) {
                    if (raw[i] < 0) continue;
                    auto span = Token::decode_date(raw_time[i]);
                    if (span.start_date <= 0) {
                        error("BAD RECORD {}: {} - {}", raw[i], span.start_date, span.end_date);
                    }
                    if (!range.overlaps(span)) continue;
                    ++cnt;
                    int32_t c = raw[i];
                    c = dict.lookup(c);
                    float const *from = &weights(c);
                    for (int j = 0; j < hidden; ++j) {
                        out[j * 2] += from[j];
                    }
                }
                if (cnt > 0) {
                    for (int j = 0; j < hidden; ++j) {
                        out[j*2+1] = out[j*2]/cnt;
                    }
                }
                out += hidden * 2;
            }
        }

        xt::xtensor<double, 2> extractBatch (py::object batch) const {
            xt::xtensor<int64_t, 2> sid = batch["patients"].cast<xt::xtensor<int64_t, 2>>();
            auto sid_shape = sid.shape();
            int n = sid_shape[0];
            xt::xtensor<double, 2> fts({n, hidden * terms.size() * 2});
            py::list records = batch["records"].cast<py::list>();
            int64_t const *sid_ptr = sid.data();
            CHECK(py::len(records) == n);
            for (int i = 0; i < n; ++i) {
                py::list one = records[i].cast<py::list>();
                double *out = &fts(i);
                vector<int64_t> raw;
                vector<int64_t> raw_time;
                for (auto h: one) {
                    Token token(h.cast<py::list>());
                    raw.push_back(token.concept);
                    raw_time.push_back(token.encode_date());
                }
                auto it = cohort.find(sid_ptr[i]);
                CHECK(it != cohort.end());
                if (sid_ptr[i] == 1446954) {
                    info("start date: {}", it->second.start_date);
                }
                embed_one(out, it->second.start_date, raw.size(), &raw[0], &raw_time[0]);
            }
            return fts;
        }

        xt::xtensor<double, 2> extractPartition (py::object prediction, string const &partition) {
            Partition part(partition);

            int rows = py::len(prediction);
            xt::xtensor<double, 2> fts({rows, hidden * terms.size() * 2});
            int off = 0;
            for (auto h: prediction.attr("iterrows")()) {
                auto row = h.cast<py::tuple>()[1];
                int64_t sid = row["subjectId"].cast<double>();
                int32_t start_date = row["cohortStartDate"].cast<double>();
                double *out = &fts(off++);
                Record const *rec = part.get_person_record(source, sid);
                if (rec == nullptr) {
                    warn("Cannot find record for {}: {}", source, sid);
                }
                else {
                    vector<int64_t> raw;
                    vector<int64_t> raw_time;
                    for (auto const &token: rec->tokens) {
                        raw.push_back(token.concept);
                        raw_time.push_back(token.encode_date());
                    }
                    embed_one(out, start_date, raw.size(), &raw[0], &raw_time[0]);
                }
            }
            return fts;
        }
    };
#endif

}

PYBIND11_MODULE(omop_core, module)
{
    using namespace omop;
    Token::test_encode();
    module.doc() = "";
    module.def("build_dict", &build_dict);
    module.attr("JULIAN_EPOCH") = 2440588;
    module.def("julian_to_string", &julian_to_string);
    module.def("test_julian", []() {
            py::list v;
            v.append(parse_date("1970-01-01"));
            v.append(parse_date("2020-02-28"));
            v.append(parse_date("2200-12-31"));
            return v;
    });


    py::class_<PartitionLoader>(module, "PartitionLoader")
         .def(py::init<int>())
         .def("load_file", &PartitionLoader::load_file)
         .def("save", &PartitionLoader::save)
         ;

    py::class_<loaders::BertDict>(module, "BertDict")
         .def(py::init<py::dict>())
         .def("dict", &loaders::BertDict::dict)
         .def("special", &loaders::BertDict::special)
         ;

#if 0
    py::class_<Encoder>(module, "Encoder")
        .def(py::init<py::dict>())
        .def("updateCohort", &Encoder::updateCohort)
        .def("trainBatch", &Encoder::trainBatch)
        .def("finishTrain", &Encoder::finishTrain)
        .def("extractBatch", &Encoder::extractBatch)
        .def("extractPartition", &Encoder::extractPartition)
        ;

    py::class_<Embedder>(module, "Embedder")
        .def(py::init<py::dict, Matrix<float>>())
        .def("updateCohort", &Embedder::updateCohort)
        .def("extractBatch", &Embedder::extractBatch)
        .def("extractPartition", &Embedder::extractPartition)
        ;
#endif

    py::class_<Partition>(module, "Partition")
        .def(py::init<string const &>())
        .def("size", &Partition::size)
        .def("get", &Partition::get)
        .def("get_person", &Partition::get_person)
        ;

    py::class_<Stat>(module, "Stat")
        .def(py::init<>())
        .def("update", &Stat::update)
        .def("get", &Stat::get)
        .def("report", &Stat::report)
        ;

#if 0
    Streamer<loaders::Vanilla>::expose(module, "VanillaStreamer");
    MemoryStreamer<loaders::Vanilla>::expose(module, "VanillaMemoryStreamer");
#endif
    Streamer<loaders::Basic>::expose(module, "BasicStreamer");
    //Streamer<loaders::MLM>::expose(module, "MLMStreamer");
    SmallStreamer<loaders::Basic>::expose(module, "BasicSmallStreamer");
    MemoryStreamer<loaders::Basic>::expose(module, "BasicMemoryStreamer");
    //MemoryStreamer<loaders::MLM>::expose(module, "MLMMemoryStreamer");


}




// Copyright © 2022 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     
// http://www.apache.org/licenses/LICENSE-2.0
//     
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

