import React, { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs";

export default function CameraFeed() {
  const [poseLabel, setPoseLabel] = useState("Sin postura detectada");
  const [selectedShirt, setSelectedShirt] = useState("/camisa1.png");
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);

  const shirtImage = useRef(new Image());

  // 🔄 Cambia la imagen cuando el usuario selecciona otra
  useEffect(() => {
    shirtImage.current.src = selectedShirt;
  }, [selectedShirt]);

  useEffect(() => {
    async function init() {
      await tf.setBackend("webgl");
      await tf.ready();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });

      const video = videoRef.current;
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        video.play();
      };

      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        }
      );

      detectorRef.current = detector;
      requestAnimationFrame(detectPose);
    }

    async function detectPose() {
      if (!videoRef.current || !canvasRef.current || !detectorRef.current)
        return;

      const poses = await detectorRef.current.estimatePoses(videoRef.current);
      drawCanvas(poses);

      requestAnimationFrame(detectPose);
    }

    function drawCanvas(poses) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      canvas.width = 640;
      canvas.height = 480;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const keypointConnections = [
        ["left_shoulder", "right_shoulder"],
        ["left_shoulder", "left_elbow"],
        ["left_elbow", "left_wrist"],
        ["right_shoulder", "right_elbow"],
        ["right_elbow", "right_wrist"],
        ["left_shoulder", "left_hip"],
        ["right_shoulder", "right_hip"],
        ["left_hip", "right_hip"],
        ["left_hip", "left_knee"],
        ["left_knee", "left_ankle"],
        ["right_hip", "right_knee"],
        ["right_knee", "right_ankle"],
      ];

      poses.forEach((pose) => {
        const keypoints = {};
        pose.keypoints.forEach((keypoint) => {
          if (keypoint.score > 0.4) {
            keypoints[keypoint.name] = keypoint;

            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "lime";
            ctx.fill();
          }
        });

        keypointConnections.forEach(([p1, p2]) => {
          const point1 = keypoints[p1];
          const point2 = keypoints[p2];

          if (point1 && point2) {
            ctx.beginPath();
            ctx.moveTo(point1.x, point1.y);
            ctx.lineTo(point2.x, point2.y);
            ctx.strokeStyle = "aqua";
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        });

        const ls = keypoints["left_shoulder"];
        const rs = keypoints["right_shoulder"];
        const le = keypoints["left_elbow"];
        const re = keypoints["right_elbow"];
        const lw = keypoints["left_wrist"];
        const rw = keypoints["right_wrist"];
        const lh = keypoints["left_hip"];
        const rh = keypoints["right_hip"];

        if (ls && rs && le && re && lw && rw && lh && rh) {
          const isLeftArmHorizontal =
            Math.abs(ls.y - le.y) < 30 && Math.abs(le.y - lw.y) < 30;
          const isRightArmHorizontal =
            Math.abs(rs.y - re.y) < 30 && Math.abs(re.y - rw.y) < 30;
          const isArmsExtended = isLeftArmHorizontal && isRightArmHorizontal;

          const isLeftArmUp = lw.y < ls.y && le.y < ls.y;
          const isRightArmUp = rw.y < rs.y && re.y < rs.y;
          const areBothArmsUp = isLeftArmUp && isRightArmUp;

          const distance = (a, b) =>
            Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
          const isLeftHandOnHip = distance(lw, lh) < 80;
          const isRightHandOnHip = distance(rw, rh) < 80;
          const isHandsOnHips = isLeftHandOnHip && isRightHandOnHip;

          if (areBothArmsUp) {
            setPoseLabel("Brazos arriba");
          } else if (isArmsExtended) {
            setPoseLabel("Postura en T");
          } else if (isHandsOnHips) {
            setPoseLabel("Manos en la cintura");
          } else {
            setPoseLabel("Otra postura");
          }
        } else {
          setPoseLabel("Sin postura detectada");
        }

        const shirtPoints = [
          "left_shoulder",
          "right_shoulder",
          "left_hip",
          "right_hip",
        ];
        if (
          shirtImage.current.complete &&
          shirtPoints.every((pt) => keypoints[pt])
        ) {
          const ls = keypoints["left_shoulder"];
          const rs = keypoints["right_shoulder"];
          const lh = keypoints["left_hip"];
          const rh = keypoints["right_hip"];

          const centerX = (ls.x + rs.x + lh.x + rh.x) / 4;
          const centerY = (ls.y + rs.y + lh.y + rh.y) / 4;
          const width = Math.hypot(rs.x - ls.x, rs.y - ls.y) * 2;
          const height = Math.hypot(lh.y - ls.y, rh.y - rs.y);
          const angle = Math.atan2(rs.y - ls.y, rs.x - ls.x);

          ctx.save();
          ctx.translate(centerX, centerY);
          ctx.rotate(angle);
          ctx.scale(1, -1);
          ctx.globalAlpha = 0.85;
          ctx.shadowColor = "rgba(0, 255, 255, 0.6)";
          ctx.shadowBlur = 20;
          ctx.shadowOffsetX = 0;
          ctx.shadowOffsetY = 0;

          ctx.drawImage(
            shirtImage.current,
            -width / 2,
            -height / 2,
            width,
            height
          );

          ctx.globalAlpha = 1;
          ctx.shadowBlur = 0;
          ctx.shadowColor = "transparent";
          ctx.restore();
        }
      });
    }

    init();
  }, [selectedShirt]);

  return (
    <div className="relative w-full max-w-xl mx-auto">
      <video
        ref={videoRef}
        className="w-full h-auto rounded-md"
        autoPlay
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />
      <div className="text-center mt-4 text-xl font-semibold text-white bg-black bg-opacity-50 p-2 rounded">
        🧍 Postura detectada: {poseLabel}
      </div>

      {/* 👕 Botones para cambiar de camiseta */}
      <div className="flex justify-center gap-4 mt-6">
        <button
          onClick={() => setSelectedShirt("/camisa1.png")}
          className="bg-white text-black px-4 py-2 rounded hover:bg-gray-200"
        >
          Camiseta 1
        </button>
        <button
          onClick={() => setSelectedShirt("/camisa2.png")}
          className="bg-white text-black px-4 py-2 rounded hover:bg-gray-200"
        >
          Camiseta 2
        </button>
        <button
          onClick={() => setSelectedShirt("/camisa3.png")}
          className="bg-white text-black px-4 py-2 rounded hover:bg-gray-200"
        >
          Camiseta 3
        </button>
      </div>
    </div>
  );
}
