// framework_log. [2023_10_11] RCSZ.
// 2024_03_19: ��ֲ framework_log => spca_tool_logger RCSZ.
// update: 2024_03_22.

#ifndef _SPCA_TOOL_LOGGER_H
#define _SPCA_TOOL_LOGGER_H
#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include <cstdarg>

#include <thread>
#include <mutex>
#include <atomic>

#define MODULE_LABEL_LOGGER "SPCA_LOGGER"

// vector 3d x, y, z.
template <typename mvec>
struct Vector3T {
	mvec vector_x, vector_y, vector_z;
	constexpr Vector3T() : vector_x(0), vector_y(0), vector_z(0) {}
	constexpr Vector3T(mvec x, mvec y, mvec z) : vector_x(x), vector_y(y), vector_z(z) {}

	mvec* data() { return &vector_x; }
	const mvec* data() const { return &vector_x; }
};

// core framework logger label.
enum LOGLABEL {
	LogError   = 1 << 1, // ��ǩ <����>
	LogWarning = 1 << 2, // ��ǩ <����>
	LogInfo    = 1 << 3, // ��ǩ <��Ϣ>
	LogTrace   = 1 << 4, // ��ǩ <����>
	LogPerfmac = 1 << 5  // ��ǩ <����>
};

// format number => string, %d(fill_zero).
std::string FMT_NUMBER_FILLZERO(uint32_t number, int32_t digits);

namespace LOGCONS {
	// @param "label" (level label), "module_name" (module), "logstr_text" (log information)
	void PushLogProcess(const LOGLABEL& label, const std::string& module_name, const std::string& logstr_text);
	// @param label, module_label, text, params. [20231205]
	void PushLogger(const LOGLABEL& label, const char* module_label, const char* log_text, ...);

	// false: not printing on the console.
	void SET_PRINTLOG_STATE(bool status_flag);

	// @return Vector3T<size_t> (x : lines, y : warring, z : error)
	Vector3T<size_t> LogLinesStatistics();

	namespace ReadLogCache {
		struct LogCache {
			// lv.0: log_label, lv.1: log_model_name, lv.2: log_string.
			std::string LogString;
			std::string LogModuleName;
			LOGLABEL    LogLabel;
			LogCache(const std::string& str, const std::string& name, LOGLABEL lab) :
				LogString(str), LogModuleName(name), LogLabel(lab)
			{}
		};
		// @param  uint32_t, back - lines.
		// @return string
		std::vector<LogCache> ExtractLogSegment(const uint32_t& lines);
	}
	// get src time[nanoseconds].
	size_t GetTimeCountNow();
}

namespace LOGGER_FILESYS {
	// async thread process. write folder.
	bool StartLogFileProcess(const char* folder);
	bool FreeLogFileProcess();
}

// framework global generate_key.
class PSAG_SYSGEN_TIME_KEY {
private:
	static size_t     TimeCountBuffer;
	static std::mutex TimeCountBufferMutex;
public:
	// generate unique number(time:nanoseconds).
	size_t PsagGenTimeKey() {
		std::lock_guard<std::mutex> Lock(TimeCountBufferMutex);

		size_t GenNumberTemp = (size_t)std::chrono::duration_cast
			<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

		if (GenNumberTemp == TimeCountBuffer) {
			// prevent duplication.
			std::chrono::nanoseconds ThreadSleep(1);
			std::this_thread::sleep_for(ThreadSleep);
		}
		TimeCountBuffer = GenNumberTemp;

		return GenNumberTemp;
	}
};

#endif