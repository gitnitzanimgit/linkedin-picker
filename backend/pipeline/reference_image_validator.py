from typing import Tuple
from ..models.image import Image
from ..services.image_service import ImageService
from ..services.face_analysis_service import FaceAnalysisService


def is_good_reference(
    img: Image,
    face_analysis_service: FaceAnalysisService = FaceAnalysisService(),
    image_service: ImageService = ImageService()
) -> Tuple[bool, dict]:
    """
    Determines if an image is a good reference image based on:
      - Exactly one face
      - Detection confidence
      - Face size
      - Pose alignment
      - Sharpness
      - Brightness

    Args:
        img (Image): The input image.
        face_analysis_service (FaceAnalysisService): Handles face-related analysis.
        image_service (ImageService): Handles image-level analysis.

    Returns:
        Tuple[bool, dict]:
            - True if all conditions are met, False otherwise.
            - A dictionary with detailed diagnostic scores or a rejection reason.
    """
    details: dict[str, float | str | bool] = {}

    # Check if there is only one face
    if not face_analysis_service.has_single_face(img):
        return False, {"reason": "need_single_face"}

    face = face_analysis_service.get_faces(img)[0]

    # Check detection confidence score
    det_conf_score = face_analysis_service.get_det_conf_score(face)
    details["det_conf_score"] = det_conf_score

    is_good_face = face_analysis_service.is_face(face)
    if not is_good_face:
        return False, {"reason": "low_detection_conf", **details}

    # Check face size
    is_good_size, area_ratio = face_analysis_service.check_size_for_ref(face, img)
    details["area_ratio"] = area_ratio
    if not is_good_size:
        return False, {"reason": "bad_size", **details}

    # Check pose
    is_good_pose, (yaw, pitch, roll) = face_analysis_service.check_pose(face)
    details.update(yaw=yaw, pitch=pitch, roll=roll)
    if not is_good_pose:
        return False, {"reason": "bad_pose", **details}

    # Check sharpness
    is_sharp, sharpness_score = image_service.check_sharpness(img)
    details["sharpness_score"] = sharpness_score
    if not is_sharp:
        return False, {"reason": "too_blurry", **details}

    # Check brightness
    is_bright, brightness = image_service.check_brightness(img)
    details["brightness"] = brightness
    if not is_bright:
        return False, {"reason": "bad_lighting", **details}

    # All checks passed
    return True, details
