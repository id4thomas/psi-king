from pathlib import Path

from PIL import Image

def crop_image(file_path: Path, bbox: list[float], page_number: int = 0) -> Image.Image:
    """Crop the image based on the bounding box

    Args:
        file_path (Path): path to the image file
        bbox (list[float]): bounding box of the image (in percentage [x0, y0, x1, y1])
        page_number (int, optional): page number of the image. Defaults to 0.

    Returns:
        Image.Image: cropped image
    """
    left, upper, right, lower = bbox

    left, right = min(left, right), max(left, right)
    upper, lower = min(upper, lower), max(upper, lower)

    img: Image.Image
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz
        except ImportError:
            raise ImportError("Please install PyMuPDF: 'pip install PyMuPDF'")

        doc = fitz.open(file_path)
        page = doc.load_page(page_number)
        pm = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
    elif suffix in [".tif", ".tiff"]:
        img = Image.open(file_path)
        img.seek(page_number)
    else:
        img = Image.open(file_path)

    return img.crop(
        (
            int(left * img.width),
            int(upper * img.height),
            int(right * img.width),
            int(lower * img.height),
        )
    )