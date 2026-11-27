// Neurosonde tests: scratch folders and environment, on every platform. The Mac keeps /private/tmp (the video
// tools read the test brain from there); Windows and Linux use the system temp folder.
#pragma once
#include <filesystem>
#include <string>
#include <cstdlib>

inline std::string testDir (const std::string& name, bool clear = true)
{
   #if defined(__APPLE__)
    const std::filesystem::path p = std::filesystem::path ("/private/tmp") / name;
   #else
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
   #endif
    std::error_code ec;
    if (clear) std::filesystem::remove_all (p, ec);
    return p.string();
}

inline void setEnv (const char* key, const std::string& value)
{
   #if defined(_WIN32)
    _putenv_s (key, value.c_str());
   #else
    ::setenv (key, value.c_str(), 1);
   #endif
}
