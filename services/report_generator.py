import base64
import io
from datetime import datetime
from typing import Optional

from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def _safe_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_image_value(value) -> Optional[str]:
    if isinstance(value, dict):
        return (
            value.get("base64")
            or value.get("image")
            or value.get("image_base64")
            or value.get("data")
        )
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _decode_base64_image(image_value: Optional[str]) -> Optional[io.BytesIO]:
    if not image_value:
        return None
    if "," in image_value:
        image_value = image_value.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_value)
    except Exception:
        return None
    return io.BytesIO(image_bytes)


def _prepare_report_image_stream(image_value: Optional[str], is_segmentation: bool = False) -> Optional[io.BytesIO]:
    image_stream = _decode_base64_image(image_value)
    if image_stream is None:
        return None

    try:
        image = Image.open(image_stream).convert("RGB")
    except Exception:
        return None

    target_width = 1600 if is_segmentation else 1400
    if image.width < target_width:
        scale = target_width / float(max(image.width, 1))
        image = image.resize((int(image.width * scale), int(image.height * scale)), RESAMPLE_LANCZOS)
    else:
        image.thumbnail((target_width, 1600), RESAMPLE_LANCZOS)

    if is_segmentation:
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.SHARPEN)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return buffer


def _build_report_image(label: str, image_value: Optional[str], width: float = 5.6 * inch, is_segmentation: bool = False):
    image_stream = _prepare_report_image_stream(image_value, is_segmentation=is_segmentation)
    if image_stream is None:
        return None

    image = Image.open(image_stream)
    image_width, image_height = image.size
    aspect_ratio = image_height / float(max(image_width, 1))
    image_stream.seek(0)
    report_image = RLImage(image_stream, width=width, height=width * aspect_ratio)
    return [Paragraph(f"<b>{label}</b>", getSampleStyleSheet()["BodyText"]), report_image]


def build_medical_report_pdf(report: dict) -> io.BytesIO:
    """Generate an upgraded PDF report with enhanced and segmented images."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=48,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ClinicalTitle", parent=styles["Heading1"], fontSize=22, spaceAfter=20)
    story = []

    story.append(Paragraph("NeuroDetect AI - Unified Neurodiagnostic Report", title_style))
    story.append(Paragraph(f"Report ID: {report.get('id', 'N/A')}", styles["BodyText"]))
    if report.get("date"):
        try:
            report_date = datetime.fromisoformat(report["date"]).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            report_date = str(report["date"])
        story.append(Paragraph(f"Date: {report_date}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    patient_rows = [
        ["Patient Name", report.get("patient_name") or "N/A"],
        ["Age", report.get("patient_age") or "N/A"],
        ["Gender", report.get("patient_gender") or "N/A"],
        ["Detection Type", "Brain Tumor" if report.get("type") == "brain" else "Alzheimer's"],
    ]
    patient_table = Table(patient_rows, colWidths=[1.8 * inch, 4.3 * inch])
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(Paragraph("<b>Patient Information</b>", styles["Heading2"]))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    analysis = _safe_dict(report.get("analysis"))
    tumor = _safe_dict(analysis.get("tumor"))
    alzheimers = _safe_dict(analysis.get("alzheimers"))
    summary_rows = [
        ["Tumor Detected", str(tumor.get("detected", report.get("tumor_detected", False)))],
        ["WHO Tumor Grade", str(tumor.get("grade", report.get("tumor_grade", "N/A")) or "N/A")],
        ["Tumor Volume (mm^3)", str(tumor.get("volume_mm3", report.get("tumor_volume_mm3", 0)))],
        ["Tumor Confidence", str(tumor.get("confidence", report.get("tumor_confidence", report.get("ai_confidence", "N/A"))))],
        ["Alzheimer Detected", str(alzheimers.get("detected", report.get("alzheimer_detected", False)))],
        ["Alzheimer Stage", str(alzheimers.get("stage", report.get("alzheimer_stage", "N/A")) or "N/A")],
        ["Alzheimer Confidence", str(alzheimers.get("confidence", report.get("alzheimer_confidence", "N/A")))],
    ]
    summary_table = Table(summary_rows, colWidths=[2.2 * inch, 3.9 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ]
        )
    )
    story.append(Paragraph("<b>Clinical AI Summary</b>", styles["Heading2"]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    report_images = _safe_dict(report.get("report_images"))
    images = []
    for label, image_value, is_segmentation in (
        (
            "Input Scan",
            _safe_image_value(report.get("input_image")) or _safe_image_value(report_images.get("input_image")) or _safe_image_value(report.get("image")),
            False,
        ),
        (
            "Original MRI",
            _safe_image_value(report.get("original_mri")) or _safe_image_value(report.get("original_image")) or _safe_image_value(report_images.get("original_mri")),
            False,
        ),
        (
            "Enhanced MRI",
            _safe_image_value(report.get("enhanced_mri")) or _safe_image_value(report.get("enhanced_image")) or _safe_image_value(report_images.get("enhanced_mri")),
            False,
        ),
        (
            "Segmentation Overlay",
            _safe_image_value(report.get("segmentation_overlay")) or _safe_image_value(report.get("segmentation_image")) or _safe_image_value(report_images.get("segmentation_overlay")),
            True,
        ),
        (
            "Segmentation Mask",
            _safe_image_value(report.get("segmentation_mask")) or _safe_image_value(report_images.get("segmentation_mask")),
            True,
        ),
    ):
        image_block = _build_report_image(label, image_value, is_segmentation=is_segmentation)
        if image_block is not None:
            images.append(image_block)

    if images:
        story.append(Paragraph("<b>Imaging Outputs</b>", styles["Heading2"]))
        for label_paragraph, report_image in images:
            story.append(label_paragraph)
            story.append(Spacer(1, 4))
            story.append(report_image)
            story.append(Spacer(1, 10))

    detailed_info = _safe_dict(report.get("detailed_info"))
    if detailed_info:
        story.append(Paragraph("<b>Clinical Interpretation</b>", styles["Heading2"]))
        for label, key in (
            ("Description", "description"),
            ("Symptoms", "symptoms"),
            ("Treatment", "treatment"),
            ("Risk Factors", "risk_factors"),
            ("Prevention", "prevention"),
            ("Severity", "severity"),
            ("Urgency", "urgency"),
        ):
            if detailed_info.get(key):
                story.append(Paragraph(f"<b>{label}:</b> {detailed_info[key]}", styles["BodyText"]))
                story.append(Spacer(1, 4))

    if report.get("doctor_notes") or report.get("prescription") or report.get("follow_up"):
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>Doctor Review</b>", styles["Heading2"]))
        if report.get("doctor_notes"):
            story.append(Paragraph(f"<b>Notes:</b> {report['doctor_notes']}", styles["BodyText"]))
        if report.get("prescription"):
            story.append(Paragraph(f"<b>Prescription:</b> {report['prescription']}", styles["BodyText"]))
        if report.get("follow_up"):
            story.append(Paragraph(f"<b>Follow-up:</b> {report['follow_up']}", styles["BodyText"]))

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "<i>Disclaimer: This AI-assisted report supports clinical review and is not a substitute for specialist diagnosis.</i>",
            styles["Italic"],
        )
    )

    document.build(story)
    buffer.seek(0)
    return buffer
