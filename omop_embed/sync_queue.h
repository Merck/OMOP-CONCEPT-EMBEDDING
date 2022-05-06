
template <typename T>
class SyncQueue {
    std::condition_variable has_item;
    std::condition_variable has_slot;
    std::mutex mutex;
    std::vector<T> data;
    bool stopped;
    bool producer_has_exit;
    int read;
    int write;
    ///   _ _ _ x x x x _ _ _
    //          ^       ^
    //          read    write
    //
    //    X _ _ _ _ _ _ X X X
    //      ^           ^
    //      write       read
    //
    // empty: write == read
    // full: (write + 1) % == read
public:
    SyncQueue (size_t depth)
        : data(depth),
        stopped(false),
        producer_has_exit(false),
        read(0),
        write(0)
    {
    }

    void stop () {
        std::unique_lock<std::mutex> lock(mutex);
        stopped = true;
        has_slot.notify_all();
        has_item.notify_all();
    }

    void clear (std::function<void(T)> cb) {
        std::unique_lock<std::mutex> lock(mutex);
        CHECK(stopped);
        while (read != write) {
            cb(data[read]);
            read = (read + 1) % data.size();
        }
    }

    void enqueue () {
        std::unique_lock<std::mutex> lock(mutex);
        producer_has_exit = true;
        has_item.notify_all();
    }

    bool enqueue (T v) {
        std::unique_lock<std::mutex> lock(mutex);
        int next;
        for (;;) {
            if (stopped) return false;
            next = (write + 1) % data.size();
            if (next != read) break;
            // full
            has_slot.wait(lock);
        }
        data[write] = v;
        write = next;
        has_item.notify_one();
        return true;
    }

    bool dequeue (T *v) {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            if (stopped) return false;
            if (read != write) break;
            if (producer_has_exit) return false;
            // full
            has_item.wait(lock);
        }
        *v = data[read];
        read = (read + 1) % data.size();
        has_slot.notify_one();
        return true;
    }

};




// Copyright © 2022 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     
//     http://www.apache.org/licenses/LICENSE-2.0
//     
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
//     limitations under the License.
