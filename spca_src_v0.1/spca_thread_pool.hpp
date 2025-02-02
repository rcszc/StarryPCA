// spca_thread_pool. RCSZ 2024.03.27
// EasyWorksPool(改), https://github.com/rcszc/EasyWorksPool
// Source: RCSZ 2023_08_28, Modify: RCSZ 2023_11_08.
// update: 2024.03.27

#ifndef _SPCA_THREAD_POOL_HPP
#define _SPCA_THREAD_POOL_HPP
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

#include "spca_system_tool/spca_tool_logger.hpp"

#define MODULE_LABEL_THDPOOL "SPCA_TASKS"

// run_time_type_information.
struct SpcaRttiObject {
    std::string ObjectName;
    std::size_t ObjectHash;
};

// SPCA thread(task) pool.
namespace SpcaTasks {
    // get this_thread hash unique_id.
    size_t ThisThreadID();

    namespace Error {
        typedef const char* ERRINFO;

        // 异常处理. std::exception::what, std::exception::name
        class TPerror :public std::exception {
        protected:
            std::string ErrorMessage = {};
            std::string ErrorCompName = {};
        public:
            TPerror(ERRINFO message, size_t pid, ERRINFO compname);

            ERRINFO what() const noexcept override { return ErrorMessage.c_str(); }
            ERRINFO name() const noexcept { return ErrorCompName.c_str(); }
        };
    }

    template<typename nClass>
    inline SpcaRttiObject _OBJECT_RTTI(nClass TASK_OBJ) {
        SpcaRttiObject ReturnInfo = {};

        const std::type_info& InfoGet = typeid(TASK_OBJ);

        ReturnInfo.ObjectName = InfoGet.name();
        ReturnInfo.ObjectHash = InfoGet.hash_code();

        return ReturnInfo;
    }

    class ThreadTasks {
    protected:
        std::vector<std::thread>          ThreadWorkers;
        std::queue<std::function<void()>> PoolTasks;
        std::mutex                        PoolMutex;
        std::condition_variable           WorkersCondition;
        std::atomic_uint32_t              WorkingThreadsCount{NULL};

        void ThreadsTaskExecution(uint32_t workers_num);
        void ThreadsTaskFree();

        bool PauseFlag = false;
        // current creation object_info.
        SpcaRttiObject OBJECT_INFO = {};

    public:
        ThreadTasks(uint32_t init_workers) {
            ThreadsTaskExecution(init_workers);
            PSAG_LOGGER::PushLogger(LogInfo, MODULE_LABEL_THDPOOL, "create thread_pool workers: %u", init_workers);
        };
        ~ThreadTasks() {
            ThreadsTaskFree();
            PSAG_LOGGER::PushLogger(LogInfo, MODULE_LABEL_THDPOOL, "close(free) thread_pool workers.");
        };

        // thread_pool: push => tasks queue.
        template<typename InClass, typename... ArgsParam>
        std::future<std::shared_ptr<InClass>> PushTask(ArgsParam... Args) {
            auto TaskObject = std::make_shared<std::packaged_task<std::shared_ptr<InClass>()>>(
                [Args = std::make_tuple(std::forward<ArgsParam>(Args)...), this]() mutable {
                    try {
                        return std::apply([](auto&&... Args) {
                            return std::make_shared<InClass>(std::forward<decltype(Args)>(Args)...);
                            }, std::move(Args));
                    }
                    catch (...) {
                        throw Error::TPerror("failed create object.", ThisThreadID(), "CREATE_OBJ");
                        return std::shared_ptr<InClass>(nullptr);
                    }
                });
            // create object => get object_info.
            OBJECT_INFO = _OBJECT_RTTI(TaskObject);

            std::future<std::shared_ptr<InClass>> ResultAsync = TaskObject->get_future();
            {
                std::unique_lock<std::mutex> Lock(PoolMutex);
                if (PauseFlag) {
                    // disable push task.
                    throw Error::TPerror("failed thread pool stop.", ThisThreadID(), "CREATE_OBJ");
                    return ResultAsync;
                }
                else
                    PoolTasks.emplace([TaskObject]() { (*TaskObject)(); });
            }
            WorkersCondition.notify_one();

            return ResultAsync;
        }

        SpcaRttiObject GetCreateObjectInfo() {
            std::unique_lock<std::mutex> Lock(PoolMutex);
            return OBJECT_INFO;
        }

        uint32_t GetWorkingThreadsCount();
        uint32_t GetTaskQueueCount();
        void     ResizeWorkers(uint32_t resize);
    };
}

#endif