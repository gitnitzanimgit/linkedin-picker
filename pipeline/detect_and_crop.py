from services.face_analysis_service import FaceAnalysisService
from services.cropping_service import CroppingService
from services.image_service import ImageService
from models.image import Image
from dotenv import load_dotenv
import os
import uuid

# Load environment variables
load_dotenv()
PAD_SIDE_FACTOR_LINKEDIN = float(os.getenv("PAD_SIDE_FACTOR_LINKEDIN"))
PAD_TOP_FACTOR_LINKEDIN = float(os.getenv("PAD_TOP_FACTOR_LINKEDIN"))
PAD_BOTTOM_FACTOR_LINKEDIN = float(os.getenv("PAD_BOTTOM_FACTOR_LINKEDIN"))
print(os.getenv("OUTPUT_IMG_EXT"))



def detect_and_crop(
    ref_img: Image,
    gallery: list[Image],
    face_analysis_service: FaceAnalysisService = FaceAnalysisService(),
    cropping_service: CroppingService = CroppingService(),
    image_service: ImageService = ImageService(),
    cropped_dir: str = os.getenv("CROPPED_DIR_PATH"),
    ext: str = os.getenv("OUTPUT_IMG_EXT")
) -> list[Image]:
    # More robust handling for the output file extension.
    # This prevents crashes if OUTPUT_IMG_EXT is present but empty in the .env file.
    output_ext = ext or ".jpg"

    # Validate number of faces
    num_faces = face_analysis_service.num_faces(ref_img)
    if num_faces != 1:
        raise ValueError("Reference image is supposed to have only one face")

    # Extract face from reference image
    ref_face = face_analysis_service.get_faces(ref_img)[0]

    detected_cropped_gallery = []
    for img in gallery:
        # Extract faces from image
        img_faces = face_analysis_service.get_faces(img)

        for face in img_faces:
            if face_analysis_service.is_same_person(ref_face, face):
                # Crop the image to LinkedIn format
                new_pixels = cropping_service.crop_image_around_face(img, face, pad_side_factor=PAD_SIDE_FACTOR_LINKEDIN,
                                                                     pad_top_factor=PAD_TOP_FACTOR_LINKEDIN,
                                                                     pad_bottom_factor=PAD_BOTTOM_FACTOR_LINKEDIN)

                # Create a new Image object with cropped pixels and unique path
                new_img = image_service.create_image(new_pixels)
                new_img.original_pixels = img.original_pixels  # Preserve original pixels through pipeline

                # --- deterministic alternative to a 2nd face-detector pass ---
                # Keep only images that have only one face post crop
                if face_analysis_service.has_single_face(new_img):
                    new_img.path = f"{cropped_dir}/{uuid.uuid1()}{output_ext}"
                    detected_cropped_gallery.append(new_img)

    return detected_cropped_gallery
