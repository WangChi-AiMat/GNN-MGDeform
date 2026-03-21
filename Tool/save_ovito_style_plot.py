import os
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, AssignColorModifier,DeleteSelectedModifier,InvertSelectionModifier
from ovito.vis import Viewport, TachyonRenderer


def save_ovito_style_plot(sample_path, center_atom_idx, important_atom_indices, output_file):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_png = os.path.join(output_file, f"NewCluster_ID{center_atom_idx}.png")

    pipeline = import_file(sample_path, sort_particles=True)
    pipeline.add_to_scene()

    topk_expr = f"ParticleIdentifier == {center_atom_idx + 1} ||" + " || ".join([f"ParticleIdentifier == {i+1}" for i in important_atom_indices])

    pipeline.modifiers.append(ExpressionSelectionModifier(expression=topk_expr))


    pipeline.modifiers.append(InvertSelectionModifier(operate_on="particles"))

    pipeline.modifiers.append(DeleteSelectedModifier(operate_on={"particles"}))

    viewport = Viewport(type=Viewport.Type.Front)
    viewport.zoom_all()


    viewport.render_image(
        filename=results_png,
        size=(900, 600),
        renderer=TachyonRenderer()
    )

    export_file(pipeline, f"{results_png}.xyz", "xyz", columns=
    ["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])

    pipeline.remove_from_scene()
    print(f"{results_png}")
