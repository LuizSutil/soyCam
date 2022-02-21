from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import csv
import time
import datetime
from scipy.spatial import distance
import math


# # Args
# parser = argparse.ArgumentParser()
# parser.add_argument("-pcl", "--pointcloud", help="enables point cloud convertion and visualization", default=False, action="store_true")
# parser.add_argument("-static", "--static_frames", default=False, action="store_true",
#                     help="Run stereo on static frames passed from host 'dataset' folder")
# args = parser.parse_args()

# point_cloud = args.pointcloud

# altura da camera em x e y
areaCamH = 143
areaCamV = 123
# pixels na camera
pixelsH = 1280
pixelsV = 800
#
countArea = 45
previsaoInicial = 15
tresholdDetectionX = 16
tresholdDetectionY = 30
deslocInicialH = previsaoInicial / areaCamV
threshX = tresholdDetectionX / areaCamH
threshY = tresholdDetectionY / areaCamV

startLine = (countArea) / areaCamV

counter = 0
frame_count = 0
listaGraos = []
lastId = 0

## Variaveis da camera
# Exposição
expTime = 950
# ISO
sensIso = 200

## Setup Luxonis
# Inicia o Objeto da Pipeline
pipeline = dai.Pipeline()

# Cria as pipeline
monoRight = pipeline.create(dai.node.MonoCamera)
manip = pipeline.create(dai.node.ImageManip)
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
manipOut = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)
controlIn = pipeline.create(dai.node.XLinkIn)
configIn = pipeline.create(dai.node.XLinkIn)

# Propriedades da Camera Mono (Direita)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

# Configurações da imagem da camera
manip.initialConfig.setResize(300, 300)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
monoRight.setFps(30)

# Configurações da NN
nn.setBlobPath(str((Path(__file__).parent / Path('./SojaModelV1.blob')).resolve().absolute()))
nn.setConfidenceThreshold(0.2)
# nn.setNumInferenceThreads(1)
# nn.setNumPoolFrames(2)

# Linking dos inputs/outputs
controlIn.out.link(monoRight.inputControl)
monoRight.out.link(manip.inputImage)
manip.out.link(nn.input)
manip.out.link(manipOut.input)
nn.out.link(nnOut.input)

# Cria as streams
manipOut.setStreamName("right")
nnOut.setStreamName("nn")
controlIn.setStreamName('control')
configIn.setStreamName('config')


# Functions
# Normalize uma bounding box para o a frame do CV
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def clamp(num, v0, v1):
    return max(v0, min(num, v1))


class Detection:
    def __init__(self, frame, xmin, xmax, ymin, ymax):
        self.frame = frame
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax


# Classes
class GraoId:
    def __init__(self, idvalue, graodetections):
        self.idvalue = idvalue
        self.graodetections = graodetections


class GraoDetection:
    def __init__(self, xmin, ymin, xmax, ymax, status, qualifications):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.status = status
        self.qualifications = qualifications


class Qualification:
    def __init__(self, xmin, ymin, xmax, ymax, nota):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.nota = nota


def displayFrame(name, frame, framecount):
    color = (255, 0, 0)

    for detection in detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        # cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
        #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
    # Show the frame
    # if len(detections) > 0:
    #     now = time.time_ns()
    #     string_i_want = str(now)
    #     cv2.imwrite(f"images/{framecount}-{string_i_want}.png", frame)
    cv2.imshow(name, frame)

with dai.Device(pipeline) as device:
    qRight = device.getOutputQueue("right", maxSize=1000, blocking=True)
    qDet = device.getOutputQueue("nn", maxSize=1000, blocking=True)
    controlQueue = device.getInputQueue(controlIn.getStreamName())

    frame = None
    detections = []

    # Controle Camera
    ctrl = dai.CameraControl()
    ctrl.setManualExposure(expTime, sensIso)

    ## fps
    startTime = time.monotonic()
    fps = 0
    ##

    #### ajustar camera
    ctrl = dai.CameraControl()
    ctrl.setManualExposure(expTime, sensIso)
    controlQueue.send(ctrl)

    while True:
        filtered = []

        # Pego proximo item queue
        inRight = qRight.get()
        inDet = qDet.get()

        # get fps
        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        #   pega ultima frame da queue
        if inRight is not None:
            frame = inRight.getCvFrame()
        #   pega ultima detects da queue
        if inDet is not None:
            detections = inDet.detections
        ##########
        frame_count += 1

        for detection in detections:
            _detection = Detection(frame_count, detection.xmin, detection.xmax, detection.ymin, detection.ymax)
            filtered.append(_detection)

        for graoitem in listaGraos:
            _lastGrao = graoitem.graodetections[len(graoitem.graodetections) - 1]
            _xGrao = _lastGrao.xmin
            _yGrao = _lastGrao.ymin

            qualified = []
            if _lastGrao.status == 1:
                for coord in filtered:
                    if (_yGrao + deslocInicialH + threshY) > coord.ymin > (_yGrao + deslocInicialH - threshY/0.8):
                        if (_xGrao + threshX) > coord.xmin > (_xGrao - threshX):
                            nota = distance.euclidean((coord.xmin, coord.ymin), (_xGrao, (_yGrao + deslocInicialH),))
                            _qualif = Qualification(coord.xmin, coord.ymin, coord.xmax, coord.ymax, nota)
                            qualified.append(_qualif)
            if _lastGrao.status > 1:
                _secondLastGrao = graoitem.graodetections[len(graoitem.graodetections) - 2]
                prevX = _xGrao + (_xGrao + _secondLastGrao.xmin)
                prevY = _yGrao + (_yGrao + _secondLastGrao.ymin)
                for coord in filtered:
                    if (prevY + threshY) > coord.ymin > (prevY - threshY):
                        nota = distance.euclidean((coord.xmin, coord.ymin), (prevX, prevY))
                        _qualif = Qualification(coord.xmin, coord.ymin, coord.xmax, coord.ymax, nota)
                        qualified.append(_qualif)
            _lastGrao.qualifications.append(qualified)

        _coordsVolatile = filtered
        usedIndex = []

        for coord in _coordsVolatile:

            if _coordsVolatile.index(coord) in usedIndex:
                continue
            else:
                melhorNota = 100

                # index no listagraos, status no index
                melhorQ = []

                for graoitem in listaGraos:
                    if len(graoitem.graodetections[len(graoitem.graodetections) - 1].qualifications) > 0:
                        _lastGraoGrades = graoitem.graodetections[len(graoitem.graodetections) - 1].qualifications[0]
                        for qual in _lastGraoGrades:
                            # if len(qual) > 0:
                            if coord.xmin == qual.xmin and coord.ymin == qual.ymin:
                                if qual.nota < melhorNota:
                                    melhorNota = qual.nota
                                    melhorQ = [listaGraos.index(graoitem),
                                               graoitem.graodetections[len(graoitem.graodetections) - 1].status]

                if len(melhorQ) == 2:
                    listaGraos[melhorQ[0]].graodetections.append(
                        GraoDetection(coord.xmin, coord.ymin, coord.xmax, coord.ymax, (melhorQ[1] + 1), [])
                    )
                    usedIndex.append(_coordsVolatile.index(coord))

        newCoords = []

        for indexNum in range(len(_coordsVolatile)):
            # remove by index
            if _coordsVolatile.index(_coordsVolatile[indexNum]) not in usedIndex:
                newCoords.append(_coordsVolatile[indexNum])

        for coord in newCoords:
            lastId += 1
            if coord.ymin < startLine:
                _newDet = GraoDetection(coord.xmin, coord.ymin, coord.xmax, coord.ymax, 1, [])
                _newGrao = GraoId(lastId, [_newDet])
                listaGraos.append(_newGrao)

        if len(filtered)> 0:
            with open('./count.txt', 'w') as writer:
                writer.write(str(len(listaGraos)))
            print()

        if cv2.waitKey(1) == ord('r'):
            listaGraos = []

        if cv2.waitKey(1) == ord('q'):
            break

        # displayFrame("preview", frame, frame_count)
