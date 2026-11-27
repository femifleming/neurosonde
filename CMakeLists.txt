cmake_minimum_required(VERSION 3.22)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "")
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "")
project(Neurosonde VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NS_COPY_AFTER_BUILD "Install into ~/Library/Audio/Plug-Ins after building" OFF)

# JUCE: -DJUCE_DIR=/path/to/JUCE, or a JUCE checkout next to this repo's parent at ~/faltung/JUCE, or fetched (9.0.5)
set(JUCE_DIR "" CACHE PATH "a JUCE checkout (leave empty to fetch JUCE 9.0.5)")
if(NOT JUCE_DIR AND EXISTS "$ENV{HOME}/faltung/JUCE/CMakeLists.txt")
    set(JUCE_DIR "$ENV{HOME}/faltung/JUCE")
endif()
if(JUCE_DIR)
    add_subdirectory(${JUCE_DIR} ${CMAKE_BINARY_DIR}/JUCE)
else()
    include(FetchContent)
    FetchContent_Declare(JUCE GIT_REPOSITORY https://github.com/juce-framework/JUCE.git GIT_TAG 9.0.5 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(JUCE)
endif()

juce_add_binary_data(NeurosondeData SOURCES
    Resources/fonts/SairaSemiCondensed-Regular.ttf Resources/fonts/SairaSemiCondensed-Medium.ttf
    Resources/fonts/SairaSemiCondensed-SemiBold.ttf Resources/fonts/ShareTechMono-Regular.ttf
    Resources/fonts/IBMPlexMono-Regular.ttf Resources/fonts/IBMPlexMono-Medium.ttf
    Resources/fonts/IBMPlexMono-Light.ttf Resources/fonts/Inter-Regular.ttf Resources/fonts/Inter-Medium.ttf)

set(NS_DEFS JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0 JUCE_DISPLAY_SPLASH_SCREEN=0 ACCELERATE_NEW_LAPACK=1)

juce_add_plugin(Neurosonde
    VERSION 1.0.0
    COMPANY_NAME "Neurosonde"
    BUNDLE_ID "com.neurosonde.neurosonde"
    PLUGIN_MANUFACTURER_CODE Frkp
    PLUGIN_CODE Nsnd
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "Neurosonde"
    ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icon/icon.png"
    DESCRIPTION "Neural resynthesis effect that learns from what it hears"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    MICROPHONE_PERMISSION_ENABLED TRUE
    MICROPHONE_PERMISSION_TEXT "Neurosonde listens to your input to learn from it and resynthesise it."
    AU_MAIN_TYPE kAudioUnitType_Effect
    VST3_CATEGORIES Fx
    COPY_PLUGIN_AFTER_BUILD ${NS_COPY_AFTER_BUILD})
target_sources(Neurosonde PRIVATE Source/Plugin.cpp Source/Editor.cpp)
target_include_directories(Neurosonde PRIVATE Source)
target_compile_definitions(Neurosonde PUBLIC ${NS_DEFS})
target_link_libraries(Neurosonde PRIVATE NeurosondeData juce::juce_audio_utils juce::juce_dsp "-framework Accelerate"
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags)

# tests: core checks + offline render + screenshots of every page
juce_add_console_app(NeuroTest PRODUCT_NAME "NeuroTest")
target_sources(NeuroTest PRIVATE tests/NeuroTest.cpp Source/Plugin.cpp Source/Editor.cpp)
target_include_directories(NeuroTest PRIVATE Source)
target_compile_definitions(NeuroTest PRIVATE ${NS_DEFS} JucePlugin_Name="Neurosonde" JucePlugin_Build_Standalone=1
    JucePlugin_IsSynth=0 JucePlugin_WantsMidiInput=0 JucePlugin_ProducesMidiOutput=0 JucePlugin_IsMidiEffect=0
    JucePlugin_ManufacturerCode=0x46726b70 JucePlugin_PluginCode=0x4e736e64 JucePlugin_VersionString="1.0.0"
    JucePlugin_Manufacturer="Neurosonde" JucePlugin_Desc="Neurosonde" JucePlugin_VersionCode=0x10000 JucePlugin_MaxNumInputChannels=2 JucePlugin_MaxNumOutputChannels=2)
target_link_libraries(NeuroTest PRIVATE NeurosondeData juce::juce_audio_utils juce::juce_dsp "-framework Accelerate" juce::juce_recommended_config_flags)

# Neurosonde Space: the same instrument, rendered into space (AmbiX to 7th order, binaural, speakers)
juce_add_plugin(NeurosondeSpace
    VERSION 1.0.0
    COMPANY_NAME "Neurosonde"
    BUNDLE_ID "com.neurosonde.neurosondespace"
    PLUGIN_MANUFACTURER_CODE Frkp
    PLUGIN_CODE Nssp
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "Neurosonde Space"
    ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icon/icon-space.png"
    DESCRIPTION "Neural resynthesis in space: learns the stereo image of what it hears"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    MICROPHONE_PERMISSION_ENABLED TRUE
    MICROPHONE_PERMISSION_TEXT "Neurosonde Space listens to your input to learn from it and place it in space."
    AU_MAIN_TYPE kAudioUnitType_Effect
    VST3_CATEGORIES Fx Spatial
    COPY_PLUGIN_AFTER_BUILD ${NS_COPY_AFTER_BUILD})
target_sources(NeurosondeSpace PRIVATE Source/Plugin.cpp Source/Editor.cpp)
target_include_directories(NeurosondeSpace PRIVATE Source)
target_compile_definitions(NeurosondeSpace PUBLIC ${NS_DEFS} NS_SPATIAL=1)
target_link_libraries(NeurosondeSpace PRIVATE NeurosondeData juce::juce_audio_utils juce::juce_dsp "-framework Accelerate"
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags)

juce_add_console_app(NeuroSpaceTest PRODUCT_NAME "NeuroSpaceTest")
target_sources(NeuroSpaceTest PRIVATE tests/NeuroTest.cpp Source/Plugin.cpp Source/Editor.cpp)
target_include_directories(NeuroSpaceTest PRIVATE Source)
target_compile_definitions(NeuroSpaceTest PRIVATE ${NS_DEFS} NS_SPATIAL=1 JucePlugin_Name="Neurosonde Space" JucePlugin_Build_Standalone=1
    JucePlugin_IsSynth=0 JucePlugin_WantsMidiInput=0 JucePlugin_ProducesMidiOutput=0 JucePlugin_IsMidiEffect=0
    JucePlugin_ManufacturerCode=0x46726b70 JucePlugin_PluginCode=0x4e737370 JucePlugin_VersionString="1.0.0"
    JucePlugin_Manufacturer="Neurosonde" JucePlugin_Desc="Neurosonde Space" JucePlugin_VersionCode=0x10000 JucePlugin_MaxNumInputChannels=2 JucePlugin_MaxNumOutputChannels=64)
target_link_libraries(NeuroSpaceTest PRIVATE NeurosondeData juce::juce_audio_utils juce::juce_dsp "-framework Accelerate" juce::juce_recommended_config_flags)

# HostCheck: loads the installed VST3 / AU like a DAW and validates them (tests/HostCheck.cpp)
juce_add_console_app(HostCheck PRODUCT_NAME "HostCheck")
target_sources(HostCheck PRIVATE tests/HostCheck.cpp)
target_compile_definitions(HostCheck PRIVATE JUCE_PLUGINHOST_VST3=1 JUCE_PLUGINHOST_AU=1 JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_MODAL_LOOPS_PERMITTED=1)
target_link_libraries(HostCheck PRIVATE juce::juce_audio_utils juce::juce_audio_processors juce::juce_recommended_config_flags)
