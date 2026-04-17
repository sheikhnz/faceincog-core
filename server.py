import argparse
import asyncio
import json
import os
from weakref import WeakSet

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from config import Config, DrawMode
from processing.detector import FaceDetector
from processing.parser import LandmarkParser
from rendering.overlay import OverlayRenderer
from masks.registry import MaskRegistry


ROOT = os.path.dirname(__file__)


class FaceIncogTrack(VideoStreamTrack):
    """
    A WebRTC video track that receives frames from a webcam (via browser),
    processes them through the FaceIncog deepfake pipeline, and returns
    the manipulated frames in real-time.
    """

    kind = "video"

    def __init__(self, track, config: Config, registry: MaskRegistry, detector: FaceDetector):
        super().__init__()
        self.track = track
        self.config = config
        self.registry = registry
        self.detector = detector
        self.parser = None
        self.renderer = None

    def process_frame(self, img):
        if self.parser is None:
            from processing.parser import LandmarkParser
            from rendering.overlay import OverlayRenderer
            self.parser = LandmarkParser(img.shape[1], img.shape[0])
            self.renderer = OverlayRenderer(draw_mode=self.config.draw_mode)

        # FaceIncog Pipeline
        raw_landmarks = self.detector.detect(img)
        face_data_list = self.parser.parse_all(raw_landmarks)
        out = self.renderer.draw(img, face_data_list, self.registry.active_mask)
        return out

    async def recv(self):
        # Retrieve incoming frame from the browser
        frame = await self.track.recv()

        try:
            # Convert to OpenCV compatible BGR numpy array
            img = frame.to_ndarray(format="bgr24")

            # Offload heavy AI processing to a separate thread 
            # so we don't freeze the aiortc event loop
            out = await asyncio.to_thread(self.process_frame, img)

            # Re-encode frame for WebRTC transmission
            new_frame = VideoFrame.from_ndarray(out, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        except Exception as e:
            print(f"[FaceIncogTrack] Error processing frame: {e}")
            # If it fails, just return the exact same frame we got so it doesn't freeze
            return frame


async def index(request):
    """Serves the frontend static HTML tester page."""
    content = open(os.path.join(ROOT, "static", "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def offer(request):
    """Accepts the SDP offer from the browser and establishes PeerConnection."""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % id(pc)
    print(f"[{pc_id}] Created")

    # Keep track of PCs to close them cleanly later
    request.app["pcs"].add(pc)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            # A simple way to control the mask via web!
            try:
                cmd = json.loads(message)
                if cmd.get("action") == "activate":
                    mask_name = cmd.get("mask")
                    print(f"[{pc_id}] Changing mask to {mask_name}")
                    request.app["registry"].activate(mask_name)
                elif cmd.get("action") == "deactivate":
                    request.app["registry"].deactivate()
            except Exception as e:
                pass


    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"[{pc_id}] ICE Connection State is {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            request.app["pcs"].discard(pc)

    @pc.on("track")
    def on_track(track):
        # When we receive the webcam track from client:
        print(f"[{pc_id}] Track {track.kind} received")
        if track.kind == "video":
            # Wrap it in our FaceIncog modification track
            modified_track = FaceIncogTrack(
                track=track,
                config=request.app["config"],
                registry=request.app["registry"],
                detector=request.app["detector"]
            )
            pc.addTrack(modified_track)

            @track.on("ended")
            async def on_ended():
                print(f"[{pc_id}] Track {track.kind} ended")

    # Establish connection
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


async def on_shutdown(app):
    # Close peer connections
    coros = [pc.close() for pc in app["pcs"]]
    await asyncio.gather(*coros)
    app["pcs"].clear()
    
    # Close detector safely
    if "detector" in app:
        app["detector"].close()


def prepare_faceincog(app, args):
    """Initialize the AI models identically to main.py before booting server."""
    config = Config(
        device_index=0,
        draw_mode=DrawMode.MASK_ONLY,
        active_mask=args.mask,
        masks_dir=args.masks_dir,
    )
    
    print("[FaceIncog] Loading Face Detector...")
    detector = FaceDetector(config.max_faces, config.min_detection_confidence,
                            config.min_tracking_confidence, config.refine_landmarks)
    detector.open()
    
    print(f"[FaceIncog] Loading Masks from {config.masks_dir}...")
    registry = MaskRegistry(masks_dir=config.masks_dir, smooth_alpha=config.mask_smooth_alpha)
    registry.load_all()
    
    if config.active_mask:
        try:
            registry.activate(config.active_mask)
            print(f"[FaceIncog] Activated default mask: {config.active_mask}")
        except Exception as e:
            print(f"[FaceIncog] Could not activate mask: {e}")

    app["config"] = config
    app["registry"] = registry
    app["detector"] = detector
    print("[FaceIncog] Subsystems ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC Server for FaceIncog")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8080, help="Port")
    parser.add_argument("--mask", type=str, default="demo_deepfake", help="Default mask")
    parser.add_argument("--masks-dir", type=str, default="assets/masks", help="Masks dir")
    args = parser.parse_args()

    app = web.Application()
    app["pcs"] = WeakSet()
    
    # Routes
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    
    # Init Pipeline Models
    prepare_faceincog(app, args)
    
    app.on_shutdown.append(on_shutdown)

    print(f"Starting server on http://{args.host}:{args.port}")
    web.run_app(app, access_log=None, host=args.host, port=args.port)
