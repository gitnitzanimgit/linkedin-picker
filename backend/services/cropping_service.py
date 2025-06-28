import numpy as np
from ..models.image import Image
from ..models.face import Face
from .face_service import FaceService
from .image_service import ImageService


class CroppingService:
    def __init__(self):
        self.face_service = FaceService()
        self.image_service = ImageService()
        self.pad_side_factor_linkedin = 0.7
        self.pad_top_factor_linkedin = 0.4
        self.pad_bottom_factor_linkedin = 1


    def get_padding_sizes(self, face: Face, pad_side_factor, pad_top_factor, pad_bottom_factor):
        # Get bbox height and width
        face_height, face_width = self.face_service.get_bbox_dimensions(face)

        # Calculate padding sizes
        pad_side_size = round(pad_side_factor * face_width)
        pad_top_size = round(pad_top_factor * face_height)
        pad_bottom_size = round(pad_bottom_factor * face_height)

        return pad_side_size, pad_top_size, pad_bottom_size

    def crop_image_around_face(
        self,
        img: Image,
        face: Face,
        pad_side_factor,
        pad_top_factor,
        pad_bottom_factor
    ) -> np.ndarray:

        # Get image dimensions
        height_img, width_img = self.image_service.get_image_dimensions(img)

        # Get bounding box coordinates
        left, top, right, bottom = self.face_service.get_bbox_coordinates(face)

        # Get padding sizes
        pad_side_size, pad_top_size, pad_bottom_size = self.get_padding_sizes(face,pad_side_factor, pad_top_factor, pad_bottom_factor)

        # Calculate new bounds
        bound_l = round(max(0, left - pad_side_size))
        bound_r = round(min(width_img, right + pad_side_size))
        bound_t = round(max(0, top - pad_top_size))
        bound_b = round(min(height_img, bottom + pad_bottom_size))

        # Get new pixels
        new_pixels = self.image_service.crop_pixels(img, bound_r=bound_r, bound_l=bound_l,
                                                    bound_t=bound_t, bound_b=bound_b)

        return new_pixels


