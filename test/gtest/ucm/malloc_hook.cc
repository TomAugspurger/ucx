/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucm/api/ucm.h>

#include <ucs/type/status.h>
#include <ucs/gtest/test.h>
#include <ucs/gtest/test_helpers.h>
#include <pthread.h>

extern "C" {
#include <ucs/sys/sys.h>
#include <malloc.h>
}

class malloc_hook : public ucs::test {
};

class test_thread {
public:
    test_thread(int index, int count, pthread_barrier_t *barrier)  :
        m_thread_ind(index), m_num_threads(count), m_barrier(barrier),
        m_map_size(0), m_unmap_size(0)
    {
        pthread_create(&m_thread, NULL, thread_func, reinterpret_cast<void*>(this));
    }

    ~test_thread() {
        join();
    }

    void join() {
        void *retval;
        pthread_join(m_thread, &retval);
    }

private:
    typedef std::pair<void*, void*> range;

    static void *thread_func(void *arg) {
        test_thread *self = reinterpret_cast<test_thread*>(arg);
        self->test();
        return NULL;
    }

    static void mem_event_callback(ucm_event_type_t event_type, ucm_event_t *event,
                                   void *arg)
    {
        test_thread *self = reinterpret_cast<test_thread*>(arg);
        self->mem_event(event_type, event);
    }

    bool is_ptr_in_range(void *ptr, size_t size, const std::vector<range> &ranges) {
        for (std::vector<range>::const_iterator iter = ranges.begin(); iter != ranges.end(); ++iter) {
            if ((ptr >= iter->first) && ((char*)ptr < iter->second)) {
                return true;
            }
        }
        return false;
    }

    void test();
    void mem_event(ucm_event_type_t event_type, ucm_event_t *event);

    static pthread_mutex_t   lock;
    static pthread_barrier_t barrier;

    int                m_thread_ind;
    int                m_num_threads;
    pthread_barrier_t  *m_barrier;
    pthread_t          m_thread;
    size_t             m_map_size;
    size_t             m_unmap_size;
    std::vector<range> m_map_ranges;
    std::vector<range> m_unmap_ranges;
};

pthread_mutex_t test_thread::lock = PTHREAD_MUTEX_INITIALIZER;

void test_thread::mem_event(ucm_event_type_t event_type, ucm_event_t *event)
{
    switch (event_type) {
    case UCM_EVENT_VM_MAPPED:
        m_map_ranges.push_back(range(event->vm_mapped.address,
                                     (char*)event->vm_mapped.address + event->vm_mapped.size));
        m_map_size += event->vm_mapped.size;
        break;
    case UCM_EVENT_VM_UNMAPPED:
        m_unmap_ranges.push_back(range(event->vm_unmapped.address,
                                       (char*)event->vm_unmapped.address + event->vm_unmapped.size));
        m_unmap_size += event->vm_unmapped.size;
        break;
    default:
        break;
    }
}

void test_thread::test() {
    static const size_t large_alloc_size = 40 * 1024 * 1024;
    static const size_t small_alloc_size = 10000;
    static const int small_alloc_count = 200 / ucs::test_time_multiplier();
    ucs_status_t result;
    ucs::ptr_vector<void> old_ptrs;
    ucs::ptr_vector<void> new_ptrs;
    size_t small_map_size;
    int num_ptrs_in_range;

    /* Allocate some pointers with old heap manager */
    for (unsigned i = 0; i < 10; ++i) {
        old_ptrs.push_back(malloc(10000));
    }

    m_map_ranges.reserve  ((small_alloc_count * 8 + 10) * m_num_threads);
    m_unmap_ranges.reserve((small_alloc_count * 8 + 10) * m_num_threads);

    pthread_barrier_wait(m_barrier);

    /* Install memory hooks */
    result = ucm_set_event_handler(UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED,
                                   0, mem_event_callback,
                                   reinterpret_cast<void*>(this));
    ASSERT_UCS_OK(result);

    /* Allocate small pointers with new heap manager */
    for (int i = 0; i < small_alloc_count; ++i) {
        new_ptrs.push_back(malloc(small_alloc_size));
    }
    EXPECT_GT(m_map_size, 0lu) << "thread " << m_thread_ind;
    small_map_size = m_map_size;

    num_ptrs_in_range = 0;
    for (ucs::ptr_vector<void>::const_iterator iter = new_ptrs.begin();
                    iter != new_ptrs.end(); ++iter)
    {
        if (is_ptr_in_range(*iter, small_alloc_size, m_map_ranges)) {
            ++num_ptrs_in_range;
        }
    }
    /* Need at least one ptr in the mapped ranges */
    EXPECT_GT(num_ptrs_in_range, 0) << "thread " << m_thread_ind;

    /* Allocate large chunk */
    void *ptr = malloc(large_alloc_size);
    EXPECT_GE(m_map_size, large_alloc_size + small_map_size) << "thread " << m_thread_ind;
    EXPECT_TRUE(is_ptr_in_range(ptr, large_alloc_size, m_map_ranges)) << "thread " << m_thread_ind;

    free(ptr);
    EXPECT_GE(m_unmap_size, large_alloc_size) << "thread " << m_thread_ind;
    /* coverity[pass_freed_arg] */
    EXPECT_TRUE(is_ptr_in_range(ptr, large_alloc_size, m_unmap_ranges)) << "thread " << m_thread_ind;

    void *s = strdup("test");
    free(s);

    /* Release old pointers (should not crash) */
    old_ptrs.clear();

    m_map_ranges.clear();
    m_unmap_ranges.clear();

    /* Don't release pointers before other threads exit, so they will map new memory
     * and not reuse memory from other threads.
     */
    pthread_barrier_wait(m_barrier);

    /* Release new pointers  */
    new_ptrs.clear();

    /* Call several malloc routines */
    malloc_trim(0);

    ptr = malloc(large_alloc_size);
    if (!RUNNING_ON_VALGRIND) {
        void *state = malloc_get_state();
        malloc_set_state(state);
    }
    free(ptr);

    pthread_mutex_lock(&lock);
    UCS_TEST_MESSAGE << "thread " << m_thread_ind << "/" << m_num_threads
                     << ": small mapped: " << small_map_size
                     <<  ", total mapped: " << m_map_size
                     <<  ", total unmapped: " << m_unmap_size;
    std::cout.flush();
    pthread_mutex_unlock(&lock);

    ucm_unset_event_handler(UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED,
                            mem_event_callback,
                            reinterpret_cast<void*>(this));
}

UCS_TEST_F(malloc_hook, hook) {
    static const int num_threads = 10;
    ucs::ptr_vector<test_thread> threads;
    pthread_barrier_t barrier;

    pthread_barrier_init(&barrier, NULL, num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.push_back(new test_thread(i, num_threads, &barrier));
    }


    threads.clear();
    pthread_barrier_destroy(&barrier);
}
