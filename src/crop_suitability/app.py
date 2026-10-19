"""Gradio web interface for the crop suitability and recommendation models.

    python -m crop_suitability.app
    python -m crop_suitability.app --share      # public Gradio link
"""

from __future__ import annotations

import argparse
import logging

import gradio as gr

from . import config
from .predict import CropAdvisor

logger = logging.getLogger(__name__)

# Gradio 6 moved `theme` from the Blocks constructor to launch(). Detect which
# signature this installation wants so the app runs warning-free on both.
_GRADIO_MAJOR = int(gr.__version__.split(".", 1)[0])
_THEME_AT_LAUNCH = _GRADIO_MAJOR >= 6


def _theme():
    return gr.themes.Soft()


def build_interface(advisor: CropAdvisor) -> gr.Blocks:
    """Construct the Gradio app around a loaded :class:`CropAdvisor`."""

    def check_suitability(n, p, k, temperature, humidity, ph, rainfall, crop) -> str:
        try:
            suitable, confidence = advisor.check_suitability(
                crop,
                N=n, P=p, K=k,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                rainfall=rainfall,
            )
        except (ValueError, FileNotFoundError) as exc:
            return f"Could not evaluate: {exc}"

        verdict = "suitable" if suitable else "not suitable"
        line = f"{crop.title()} is {verdict} for these conditions."
        if confidence is not None:
            line += f" (confidence {confidence:.0%})"
        return line

    def recommend_crop(n, p, k, temperature, humidity, ph, rainfall) -> str:
        try:
            ranked = advisor.rank_crops(
                N=n, P=p, K=k,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                rainfall=rainfall,
            )
        except (ValueError, FileNotFoundError) as exc:
            return f"Could not evaluate: {exc}"

        lines = [f"Best match: {ranked[0][0].title()}", "", "Ranked:"]
        lines += [f"  {crop.title():<10} {score:.0%}" for crop, score in ranked]
        return "\n".join(lines)

    def combined(n, p, k, temperature, humidity, ph, rainfall, crop) -> tuple[str, str]:
        return (
            check_suitability(n, p, k, temperature, humidity, ph, rainfall, crop),
            recommend_crop(n, p, k, temperature, humidity, ph, rainfall),
        )

    def slider(feature: str, label: str, step: float | None = None) -> gr.Slider:
        low, high, default = config.INPUT_RANGES[feature]
        kwargs = {"step": step} if step else {}
        return gr.Slider(low, high, value=default, label=label, **kwargs)

    block_kwargs: dict = {"title": "Crop Suitability & Recommendation System"}
    if not _THEME_AT_LAUNCH:
        block_kwargs["theme"] = _theme()

    with gr.Blocks(**block_kwargs) as demo:
        gr.Markdown("# Crop Suitability & Recommendation System")
        gr.Markdown(
            "Set the soil and weather conditions below. The first model judges "
            "whether a chosen crop suits them; the second ranks all crops it knows."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Soil nutrients (mg/kg)")
                n_input = slider("N", "Nitrogen (N)")
                p_input = slider("P", "Phosphorus (P)")
                k_input = slider("K", "Potassium (K)")
            with gr.Column():
                gr.Markdown("### Environment")
                temp_input = slider("temperature", "Temperature (°C)")
                humidity_input = slider("humidity", "Humidity (%)")
                ph_input = slider("ph", "Soil pH", step=0.1)
                rainfall_input = slider("rainfall", "Rainfall (mm)")

        crop_input = gr.Dropdown(
            choices=advisor.crop_categories,
            value=advisor.crop_categories[-1],
            label="Crop to test",
        )

        condition_inputs = [
            n_input, p_input, k_input, temp_input, humidity_input, ph_input, rainfall_input,
        ]

        with gr.Row():
            with gr.Column():
                check_btn = gr.Button("Check suitability", variant="primary")
                suitability_output = gr.Textbox(label="Suitability", interactive=False, lines=2)
            with gr.Column():
                recommend_btn = gr.Button("Find best crop", variant="secondary")
                recommendation_output = gr.Textbox(
                    label="Recommendation", interactive=False, lines=8
                )

        analyse_btn = gr.Button("Run both", variant="primary")

        check_btn.click(
            check_suitability,
            inputs=condition_inputs + [crop_input],
            outputs=suitability_output,
        )
        recommend_btn.click(
            recommend_crop, inputs=condition_inputs, outputs=recommendation_output
        )
        analyse_btn.click(
            combined,
            inputs=condition_inputs + [crop_input],
            outputs=[suitability_output, recommendation_output],
        )

        gr.Examples(
            examples=[
                [90, 42, 43, 25, 80, 6.5, 210, "rice"],
                [80, 45, 40, 26, 85, 6.8, 180, "jute"],
                [10, 10, 10, 15, 40, 5.0, 50, "banana"],
                [75, 50, 20, 23, 65, 6.2, 80, "maize"],
            ],
            inputs=condition_inputs + [crop_input],
        )

        gr.Markdown(
            "Predictions come from LightGBM models trained on public soil, weather "
            "and district-level yield data. Treat them as a screening aid, not "
            "agronomic advice."
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio app.")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=7860, help="port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    demo = build_interface(CropAdvisor())

    launch_kwargs: dict = {
        "share": args.share,
        "server_name": args.host,
        "server_port": args.port,
    }
    if _THEME_AT_LAUNCH:
        launch_kwargs["theme"] = _theme()

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
