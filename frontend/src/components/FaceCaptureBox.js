import { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  FormControlLabel,
  IconButton,
  Stack,
  Switch,
  Typography
} from "@mui/material";
import UploadRoundedIcon from "@mui/icons-material/UploadRounded";
import CameraAltRoundedIcon from "@mui/icons-material/CameraAltRounded";
import StopCircleRoundedIcon from "@mui/icons-material/StopCircleRounded";
import DeleteRoundedIcon from "@mui/icons-material/DeleteRounded";
import { PrimaryButton, SecondaryButton } from "@/components/CustomButtons";

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function buildSignature(canvas) {
  const signatureCanvas = document.createElement("canvas");
  signatureCanvas.width = 24;
  signatureCanvas.height = 24;

  const context = signatureCanvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(canvas, 0, 0, signatureCanvas.width, signatureCanvas.height);

  const pixels = context.getImageData(0, 0, signatureCanvas.width, signatureCanvas.height).data;
  const signature = [];

  for (let index = 0; index < pixels.length; index += 4) {
    signature.push(
      Math.round((pixels[index] * 0.299) + (pixels[index + 1] * 0.587) + (pixels[index + 2] * 0.114))
    );
  }

  return signature;
}

function hasMeaningfulChange(signature, signatures) {
  if (!signatures.length) {
    return true;
  }

  return signatures.every((existingSignature) => {
    let difference = 0;

    for (let index = 0; index < signature.length; index += 1) {
      difference += Math.abs(signature[index] - existingSignature[index]);
    }

    return difference / signature.length > 8;
  });
}

export default function FaceCaptureBox({
  maxImages = 5,
  onImagesChange
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileRef = useRef(null);
  const lastCaptureAtRef = useRef(0);
  const frameSignaturesRef = useRef([]);
  const [images, setImages] = useState([]);
  const [stream, setStream] = useState(null);
  const [autoCapture, setAutoCapture] = useState(true);

  useEffect(() => {
    onImagesChange(images);
  }, [images, onImagesChange]);

  useEffect(() => {
    if (!images.length) {
      frameSignaturesRef.current = [];
      lastCaptureAtRef.current = 0;
    }
  }, [images.length]);

  useEffect(
    () => () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    },
    [stream]
  );

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      return;
    }

    const mediaStream = await navigator.mediaDevices.getUserMedia({
      video: true
    });

    setStream(mediaStream);

    if (videoRef.current) {
      videoRef.current.srcObject = mediaStream;
      await videoRef.current.play();
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setStream(null);
  };

  const captureImage = useCallback((automatic = false) => {
    if (!videoRef.current || !canvasRef.current || images.length >= maxImages) {
      return false;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    const signature = buildSignature(canvas);

    if (automatic) {
      const now = Date.now();

      if (now - lastCaptureAtRef.current < 900) {
        return false;
      }

      if (!hasMeaningfulChange(signature, frameSignaturesRef.current)) {
        return false;
      }

      lastCaptureAtRef.current = now;
    }

    frameSignaturesRef.current = [...frameSignaturesRef.current, signature].slice(-maxImages);

    setImages((currentImages) => [
      ...currentImages,
      canvas.toDataURL("image/jpeg", 0.9)
    ]);

    return true;
  }, [images.length, maxImages]);

  useEffect(() => {
    if (!stream || !autoCapture || images.length >= maxImages) {
      return undefined;
    }

    const timer = setInterval(() => {
      captureImage(true);
    }, 1100);

    return () => clearInterval(timer);
  }, [autoCapture, captureImage, images.length, maxImages, stream]);

  const handleUpload = async (event) => {
    const files = Array.from(event.target.files || []).slice(
      0,
      maxImages - images.length
    );
    const uploadedImages = await Promise.all(files.map(fileToBase64));
    setImages((currentImages) => [...currentImages, ...uploadedImages]);
    event.target.value = "";
  };

  return (
    <Card variant="outlined">
      <CardContent>
        <Stack spacing={2}>
          <Stack
            direction={{ md: "row", xs: "column" }}
            justifyContent="space-between"
            spacing={1.5}
          >
            <Box>
              <Typography variant="h6">Face Enrollment</Typography>
              <Typography color="text.secondary" variant="body2">
                Start the camera and let auto-capture collect multiple angles for a stronger
                embedding profile.
              </Typography>
            </Box>

            <Stack direction="row" flexWrap="wrap" gap={1}>
              <Chip
                color={stream ? "success" : "default"}
                label={stream ? "Camera Ready" : "Camera Idle"}
                size="small"
              />
              <Chip
                color={images.length >= 3 ? "success" : "warning"}
                label={`${images.length}/${maxImages} samples`}
                size="small"
              />
            </Stack>
          </Stack>

          <Alert severity="info" sx={{ py: 0.5 }}>
            Capture at least 3 clear samples with slight head movement for more reliable recognition.
          </Alert>

          <Box
            sx={{
              border: "1px solid",
              borderColor: "divider",
              bgcolor: "background.default",
              p: 1.5
            }}
          >
            <video autoPlay className="camera-video" muted playsInline ref={videoRef} />
          </Box>
          <canvas hidden ref={canvasRef} />

          <Stack
            direction={{ md: "row", xs: "column" }}
            justifyContent="space-between"
            spacing={2}
            sx={{ alignItems: { md: "center", xs: "flex-start" } }}
          >
            <FormControlLabel
              control={
                <Switch
                  checked={autoCapture}
                  disabled={!stream}
                  onChange={(event) => setAutoCapture(event.target.checked)}
                />
              }
              label="Automatic sample capture"
            />

            <Typography color="text.secondary" variant="caption">
              Auto-capture skips near-duplicate frames to keep the training set cleaner.
            </Typography>
          </Stack>

          <Stack direction={{ sm: "row", xs: "column" }} spacing={1.5}>
            <PrimaryButton
              disabled={Boolean(stream)}
              startIcon={<CameraAltRoundedIcon />}
              onClick={startCamera}
            >
              Start Camera
            </PrimaryButton>
            <SecondaryButton
              disabled={!stream}
              startIcon={<StopCircleRoundedIcon />}
              onClick={stopCamera}
            >
              Stop Camera
            </SecondaryButton>
            <SecondaryButton
              disabled={!stream || images.length >= maxImages}
              onClick={() => captureImage(false)}
            >
              Capture Now
            </SecondaryButton>
            <SecondaryButton
              startIcon={<UploadRoundedIcon />}
              onClick={() => fileRef.current?.click()}
            >
              Upload Images
            </SecondaryButton>
            <SecondaryButton disabled={!images.length} onClick={() => setImages([])}>
              Clear Samples
            </SecondaryButton>
          </Stack>

          <input
            accept="image/*"
            hidden
            multiple
            ref={fileRef}
            type="file"
            onChange={handleUpload}
          />

          <Stack direction="row" flexWrap="wrap" gap={1.5}>
            {images.map((image, index) => (
              <Box key={`${index}-${image.slice(0, 20)}`} sx={{ position: "relative" }}>
                <Box
                  alt={`capture-${index + 1}`}
                  component="img"
                  src={image}
                  sx={{
                    width: 92,
                    height: 92,
                    objectFit: "cover",
                    borderRadius: 2,
                    border: "1px solid",
                    borderColor: "divider"
                  }}
                />
                <IconButton
                  color="error"
                  size="small"
                  sx={{ position: "absolute", right: -10, top: -10, bgcolor: "background.paper" }}
                  onClick={() =>
                    setImages((currentImages) =>
                      currentImages.filter((_, imageIndex) => imageIndex !== index)
                    )
                  }
                >
                  <DeleteRoundedIcon fontSize="small" />
                </IconButton>
              </Box>
            ))}
          </Stack>

          <Typography color="text.secondary" variant="caption">
            Captured {images.length} / {maxImages} images
          </Typography>
        </Stack>
      </CardContent>
    </Card>
  );
}
