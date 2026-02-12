#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <thread>
#include <cmath>
#include <fstream>
#include "json.hpp" // Great library for C++ JSON
#include <sys/inotify.h>
#include <unistd.h>

#include "PythonScriptClient.hpp"
#include "Stopwatch.hpp"
#include "filters/KalmanFilter.hpp"
#include "lrf/LrfManager.hpp"
#include "okb/OkbController.hpp"
#include "ptu/PtuController.hpp"
#include "ptu/MotionPlanner.hpp"
#include "FOVCalculator.hpp"
#include "visca/ViscaManager.hpp"
#include "visca/ViscaProtocol.hpp"
#include "cam/CameraStreamer.hpp"
#include "cam/CameraManager.hpp"
#include "web-comm/ManagementGate.hpp"

#include "stabilizers/TrajectoryStabilizer.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp> 
#include <opencv2/core/cuda.hpp>   
#include <opencv2/highgui.hpp>
#include <thread>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "SystemConfig.hpp"


using json = nlohmann::json;

void load_settings();
void proccessMain(OkbRxData& okbRxData);
bool initializeConnection(std::thread& processingThread);

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

const cv::Size outputSizeDay(1920, 1080);
const cv::Size outputSizeThermal(1024, 768);

const std::string lrfIP = "192.168.1.234";
const std::string jetsonIP = "192.168.1.80";
const std::string okbIP = "192.168.1.70";

double xPos = 0;
double yPos = 0;
int lastReceivedX = 0;
int lastReceivedY = 0;

double zoomXMultiplier = 1.0;

void syncTelemetry(const OkbRxData& rx);
void applyMotionScaling(PtuTxData& ptuTxData);
void handleSpecialModes(const OkbRxData& rx, PtuTxData& tx);
void updatePositionCommand(int screenX, int screenY, PtuTxData& tx, int mode);

std::unique_ptr<CameraManager> camManager;
std::unique_ptr<CameraStreamer> dayStreamer;
std::unique_ptr<CameraStreamer> nightStreamer;
std::unique_ptr<ViscaManager> viscaManager;
std::unique_ptr<LrfManager> lrfManager;

std::string gstDayCapPipeline = "";

std::string gstNightCapPipeline = "";

std::string gstDayWriterPipeline= "";
std::string gstDayWebWriterPipeline= "";

std::string gstNightWriterPipeline = "";
std::string gstNightWebWriterPipeline = "";

void runMainApp();

OkbTxData okbTxData;
OkbRxData okbRxData;
OkbSettingsData okbSettingsData;
OkbCombinedData okbCombinedData;

PtuRxData lastPtuRxData;
PtuController ptuController;

float lastDistance = 0.0f;
bool ptuConnected;
OkbController okbController;

ManagementGate managementGate;

int fd, wd;

FOVCalculator fovCalculator;
MotionPlanner planner(fovCalculator);

SystemConfig systemConfig;

static json lastConfig;
static bool hasLastConfig = false;

bool sectionChanged(const json& oldCfg, const json& newCfg, const std::string& key);


void load_settings() {
    try {
        std::ifstream f("../shared/configuration.json");
        if (!f.is_open()) return;

        json data = json::parse(f);

        static json lastConfig;
        static bool hasLastConfig = false;

        bool firstLoad = !hasLastConfig;

        auto sectionChanged = [&](const std::string& key) {
            if (firstLoad) return true;
            if (!lastConfig.contains(key) && !data.contains(key)) return false;
            if (lastConfig.contains(key) != data.contains(key)) return true;
            return lastConfig[key] != data[key];
        };

        /* ---------- zoomDictionary ---------- */
        if (sectionChanged("zoomDictionary")) {
            systemConfig.zoomMap.clear();
            if (data.contains("zoomDictionary")) {
                for (auto& [key, value] : data["zoomDictionary"].items()) {
                    uint16_t visca = static_cast<uint16_t>(std::stoi(key, nullptr, 0));
                    double zoom_x = value.get<double>();
                    systemConfig.zoomMap.push_back({ visca, zoom_x });
                }

                std::sort(systemConfig.zoomMap.begin(), systemConfig.zoomMap.end(),
                    [](const ZoomEntry& a, const ZoomEntry& b) {
                        return a.pos < b.pos;
                    });
            }
        }

        /* ---------- horizontalZoomFov ---------- */
        if (sectionChanged("horizontalZoomFov")) {
            systemConfig.hFovTable.clear();
            if (data.contains("horizontalZoomFov")) {
                for (auto& [key, value] : data["horizontalZoomFov"].items()) {
                    systemConfig.hFovTable.push_back({ std::stod(key), value.get<double>() });
                }

                std::sort(systemConfig.hFovTable.begin(), systemConfig.hFovTable.end(),
                    [](const ZoomFOVEntry& a, const ZoomFOVEntry& b) {
                        return a.zoomX < b.zoomX;
                    });
            }
        }

        /* ---------- verticalZoomFov ---------- */
        if (sectionChanged("verticalZoomFov")) {
            systemConfig.vFovTable.clear();
            if (data.contains("verticalZoomFov")) {
                for (auto& [key, value] : data["verticalZoomFov"].items()) {
                    systemConfig.vFovTable.push_back({ std::stod(key), value.get<double>() });
                }

                std::sort(systemConfig.vFovTable.begin(), systemConfig.vFovTable.end(),
                    [](const ZoomFOVEntry& a, const ZoomFOVEntry& b) {
                        return a.zoomX < b.zoomX;
                    });
            }
        }

        /* ---------- Day camera pipelines ---------- */
        if (sectionChanged("dayCapRTSP") ||
            sectionChanged("dayWriteRTSP") ||
            sectionChanged("dayWebWriteRTSP")) {

            gstDayCapPipeline =
                data.value("dayCapRTSP", gstDayCapPipeline);
            gstDayWriterPipeline =
                data.value("dayWriteRTSP", gstDayWriterPipeline);
            gstDayWebWriterPipeline =
                data.value("dayWebWriteRTSP", gstDayWebWriterPipeline);

            camManager->updateCameraPipeline(
                CameraManager::CameraType::DAY,
                gstDayCapPipeline,
                gstDayWriterPipeline,
                gstDayWebWriterPipeline,
                outputSizeDay
            );
        }

        /* ---------- Night camera pipelines ---------- */
        if (sectionChanged("nightCapRTSP") ||
            sectionChanged("nightWriteRTSP") ||
            sectionChanged("nightWebWriteRTSP")) {

            gstNightCapPipeline =
                data.value("nightCapRTSP", gstNightCapPipeline);
            gstNightWriterPipeline =
                data.value("nightWriteRTSP", gstNightWriterPipeline);
            gstNightWebWriterPipeline =
                data.value("nightWebWriteRTSP", gstNightWebWriterPipeline);

            camManager->updateCameraPipeline(
                CameraManager::CameraType::NIGHT,
                gstNightCapPipeline,
                gstNightWriterPipeline,
                gstNightWebWriterPipeline,
                outputSizeThermal
            );
        }

        /* ---------- Tracking aggressiveness ---------- */
        if (sectionChanged("trackAggressiveness")) {
            if (data.contains("trackAggressiveness")) {
                auto agg = data["trackAggressiveness"];
                systemConfig.trackAggressiveness.panAgressiveness = agg.value("panAgressiveness", systemConfig.trackAggressiveness.panAgressiveness);
                systemConfig.trackAggressiveness.tiltAgressiveness = agg.value("tiltAggressiveness", systemConfig.trackAggressiveness.tiltAgressiveness);
            }
        }

        /* ---------- FOV calculator ---------- */
        if (sectionChanged("zoomDictionary") ||
            sectionChanged("horizontalZoomFov") ||
            sectionChanged("verticalZoomFov")) {

            fovCalculator.initialize(
                systemConfig.zoomMap,
                systemConfig.hFovTable,
                systemConfig.vFovTable
            );
        }

        /* ---------- Commit ---------- */
        lastConfig = data;
        hasLastConfig = true;

    } catch (const std::exception& e) {
        std::cerr << "JSON Load Error: " << e.what() << std::endl;
    }
}

bool sectionChanged(const json& oldCfg, const json& newCfg, const std::string& key) {
    if (!oldCfg.contains(key) && !newCfg.contains(key)) return false;
    if (oldCfg.contains(key) != newCfg.contains(key)) return true;
    return oldCfg[key] != newCfg[key];
}


int main()
{
    camManager = std::make_unique<CameraManager>();

    dayStreamer = std::make_unique<CameraStreamer>(
    CameraType::DAY,
    "DayCam",
    "models/mobile_sam.engine");

nightStreamer = std::make_unique<CameraStreamer>(
    CameraType::NIGHT,
    "NightCam",
    "models/mobile_sam.engine");

    fovCalculator.initialize(systemConfig.zoomMap, systemConfig.hFovTable, systemConfig.vFovTable);
    viscaManager = std::make_unique<ViscaManager>(fovCalculator);

    fd = inotify_init();
    
    wd = inotify_add_watch(fd, "../shared/configuration.json", IN_CLOSE_WRITE);
    
    camManager->addCamera(CameraManager::CameraType::DAY, std::move(dayStreamer));
    camManager->addCamera(CameraManager::CameraType::NIGHT, std::move(nightStreamer));
    
    load_settings();

    camManager->startAll();
    camManager->setActiveCamera(CameraManager::CameraType::DAY);

    lrfManager = std::make_unique<LrfManager>(lrfIP, 10001);
    
    runMainApp();

    return 0;
}

void runMainApp()
{
    std::thread processingThread;
    try
    {
        if (!initializeConnection(processingThread))
        {
            std::cout << "Initialization Failed, Terminating App." << std::endl;
            return;
        }
        else
        {
            std::cout << "Initialization Successful, Starting Main Loop." << std::endl;
        }
    }
    catch (...)
    {
        std::cerr << "Exception occured in main app init" << std::endl;
    }
    
    while (true)
    {
        try
        {
            int bytesAvailable;
            ioctl(fd, FIONREAD, &bytesAvailable);

            if (bytesAvailable > 0) {
                char buffer[4096];
                
                read(fd, buffer, bytesAvailable);
                
                std::cout << "Change detected in configuration.json!" << std::endl;
                
                usleep(10000); 
                load_settings();
            }
            
            if (okbController.communicationUnit().sessionCount() == 0)
                continue;

            const auto optOkbRxData = okbController.read();
            if (!optOkbRxData)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            okbCombinedData = *optOkbRxData;
            std::visit(overloaded {
                [&](const OkbRxData& mainData) {
                    okbRxData = mainData;
                    
                    proccessMain(okbRxData);
                    camManager->broadcastTelemetry(okbRxData, okbTxData, fovCalculator);
                    managementGate.broadcast(okbRxData, okbTxData);
                    okbController.write(okbTxData);
                },
                [&](const OkbSettingsData& settingsData) {
                    okbSettingsData = settingsData;
                    std::cout << "Received Settings\n";
                },
                [](std::monostate) {
                    std::cout << "Received No MeaningfulData\n";
                }
            }, okbCombinedData.data);
            
        }
        catch (std::exception e)
        {
            std::cout << "Exception occured in main app loop: " << e.what() << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    okbController.stop();
    ptuController.stop();
}

bool initializeConnection(std::thread& processingThread)
{
    int okbRetryCount = 0;
    int ptuRetryCount = 0;

    bool okbConnected = false;

    while (true)
    {
        if (okbRetryCount < 5 && !okbConnected)
        {
            try
            {
                if (!okbController.start(jetsonIP, 5000))
                {
                    okbRetryCount++;
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }
                else
                {
                    okbRetryCount = 0;
                    okbConnected = true;
                    continue;
                }
            }
            catch (std::runtime_error e)
            {
                std::cerr << "TCP Server Start Error: " << e.what() << std::endl;
                okbRetryCount++;
                continue;
            }
        }

        if(ptuRetryCount < 5 && !ptuConnected)
        {
            try
            {
                if (!ptuController.start("/dev/ttyUSB0", LibSerial::BaudRate::BAUD_921600))
                {
                    ptuRetryCount++;
                    continue;
                }
                else
                {
                    ptuRetryCount = 0;
                    ptuConnected = true;
                    continue;
                }
            }
            catch (std::runtime_error e)
            {
                std::cerr << "PTU Start Error: " << e.what() << std::endl;
                ptuRetryCount++;
                continue;
            }
        }

        if (!okbConnected)
        {
            std::cout << "TCP Server failed to host. Terminating program." << std::endl;
            return false;
        }
        else
        {
            std::cout << "TCP Server Started Succesfully" << std::endl;
        }
        if (!ptuConnected)
        {
            std::cout << "PTU Failed To Connect. Check Serial Ports And Run Again If Unexpected. Program Will Run Regardless." << std::endl;
        }
        else
        {
            std::cout << "PTU Connection Started Succesfully" << std::endl;
        }

        break;
    }

    if (!managementGate.start("127.0.0.1", 5555)) {
        std::cout << "Warning: Management Gate failed to start on port 5555" << std::endl;
    }

    return true;
}

void proccessMain(OkbRxData& okbRxData) {
    viscaManager->update(okbRxData);
    zoomXMultiplier = viscaManager->getZoomX();
    lrfManager->update(okbRxData.lrfModeSelect == 1);

    PtuTxData ptuTxData;
    ptuTxData.parse(okbRxData);
    
    applyMotionScaling(ptuTxData);
    handleSpecialModes(okbRxData, ptuTxData);

    if (okbRxData.trigger1 == 1) { ptuTxData.azAxisMode = ptuTxData.elAxisMode = 0; }

    ptuController.write(ptuTxData);
    syncTelemetry(okbRxData);
}

void applyMotionScaling(PtuTxData& ptuTxData) {
    if (ptuTxData.azAxisMode == 1 && ptuTxData.elAxisMode == 1) {
        ptuTxData.azAxisCmd /= zoomXMultiplier;
        ptuTxData.elAxisCmd /= zoomXMultiplier;
    }
}

int prevTrackEnabled = 0;
void handleSpecialModes(const OkbRxData& rx, PtuTxData& tx) {
    if (rx.ptuModeSelect == 4) {
        updatePositionCommand(rx.ptuCmdX, rx.ptuCmdY, tx, 2);
    }
    else if (rx.ptuModeSelect == 6 && rx.isTrackEnabled) {
        auto activeCam = camManager->getActive();
        if (activeCam && activeCam->hasActiveTarget()) {
            cv::Point err = activeCam->getPixelError();
            //std::cout << "Tracking Error: (" << err.x << ", " << err.y << ")\n";
            tx.trackerAltMode = 2;
            updatePositionCommand(err.x, -1 * err.y, tx, 3);
            tx.azGyroBias = rx.azGyroBias;
            tx.elGyroBias = rx.elGyroBias;
        }
    }
    else if (prevTrackEnabled == 1 && !rx.isTrackEnabled) {
        tx.azAxisMode = 3;
        tx.elAxisMode = 3;
        tx.trackerAltMode = 2;
        tx.azAxisCmd = 0;
        tx.elAxisCmd = 0;
    }

    prevTrackEnabled = rx.isTrackEnabled;
}

void updatePositionCommand(int screenX, int screenY, PtuTxData& tx, int mode) {
    if (((lastReceivedX != screenX || lastReceivedY != screenY) && (mode == 2)) || mode == 3) {
        if (mode == 3) {
            float panK = systemConfig.trackAggressiveness.panAgressiveness;
            float tiltK = systemConfig.trackAggressiveness.tiltAgressiveness;

            double H_fovAgg = fovCalculator.HorizontalFOVToAgg(fovCalculator.m_HFOV_deg);
            double V_fovAgg = fovCalculator.VerticalFOVToAgg(fovCalculator.m_VFOV_deg);
            std::cout << "HFOV Aggressiveness: " << H_fovAgg << "\n";
            std::cout << "VFOV Aggressiveness: " << V_fovAgg << "\n";

            xPos = screenX * (panK * H_fovAgg);
            yPos = screenY * (tiltK * V_fovAgg);

        }
        else
        {
            MotionCommand move = planner.planRelativeMove(screenX, screenY, lastPtuRxData.azAxisPos, lastPtuRxData.elAxisPos, zoomXMultiplier);

            constexpr double BASE_MAX_SPEED = 60.0;
            constexpr double MIN_MAX_SPEED  = 3.0;
            double Pan_MAX_SPEED = MIN_MAX_SPEED + (BASE_MAX_SPEED - MIN_MAX_SPEED);
            double Tilt_MAX_SPEED = MIN_MAX_SPEED + (BASE_MAX_SPEED - MIN_MAX_SPEED); 

            planner.applyVelocityLimits(move, lastPtuRxData.azAxisPos, lastPtuRxData.elAxisPos, std::min(Pan_MAX_SPEED, Tilt_MAX_SPEED));
            xPos = move.pan;
            yPos = move.tilt;
            lastReceivedX = screenX; 
            lastReceivedY = screenY;
            tx.azAxisVelocityLimit = move.panVelocity;
            tx.elAxisVelocityLimit = move.tiltVelocity;
        }
    }
    tx.azAxisMode = tx.elAxisMode = mode;
    tx.azAxisCmd = xPos; tx.elAxisCmd = yPos;
}

void syncTelemetry(const OkbRxData& rx) {
    if (ptuConnected) {
        if (auto ptuRx = ptuController.read()) lastPtuRxData = *ptuRx;
        okbTxData.azAxisPos = lastPtuRxData.azAxisPos;
        okbTxData.elAxisPos = lastPtuRxData.elAxisPos;
        okbTxData.azAxisSpeed = lastPtuRxData.azAxisSpeed;
        okbTxData.elAxisSpeed = lastPtuRxData.elAxisSpeed;
        okbTxData.azCurrent = lastPtuRxData.azCurrent;
        okbTxData.elCurrent = lastPtuRxData.elCurrent;
    }

    if (lrfManager->isDataReady()) {
        lastDistance = lrfManager->getDistance();
        lrfManager->consumeData();
    }
    okbTxData.lrfValue = lastDistance;

    okbTxData.ptuModeSelect = rx.ptuModeSelect;
    okbTxData.camModeSelect = rx.camModeSelect;
    okbTxData.lrfModeSelect = rx.lrfModeSelect;
}