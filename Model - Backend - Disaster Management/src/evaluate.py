import torch, argparse
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import DisasterDataset, load_split_csv
from src.model import ModelConfig, MultiTaskEfficientNet
from src.utils import compute_metrics, ensure_dir, plot_confusion_matrix, save_json


def evaluate(a):
    ckpt = torch.load(a.checkpoint, map_location=a.device)
    cmap, smap = ckpt["class_to_idx"], ckpt["severity_to_idx"]
    ic, isev = {v:k for k,v in cmap.items()}, {v:k for k,v in smap.items()}

    model = MultiTaskEfficientNet(
        ModelConfig(len(cmap), len(smap))
    ).to(a.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df = load_split_csv(Path(a.data_dir) / "test.csv")
    dl = torch.utils.data.DataLoader(
        DisasterDataset(df, cmap, smap, a.image_size, "test"),
        batch_size=a.batch_size, shuffle=False
    )

    tc, pc, ts, ps = [], [], [], []

    with torch.no_grad():
        for b in dl:
            out = model(b["image"].to(a.device))
            pc += out["class_logits"].argmax(1).cpu().tolist()
            ps += out["severity_logits"].argmax(1).cpu().tolist()
            tc += b["class_label"].tolist()
            ts += b["severity_label"].tolist()

    res = {
        "disaster_metrics": compute_metrics(tc, pc),
        "severity_metrics": compute_metrics(ts, ps),
        "disaster_report": classification_report(tc, pc, target_names=[ic[i] for i in ic], output_dict=True, zero_division=0),
        "severity_report": classification_report(ts, ps, target_names=[isev[i] for i in isev], output_dict=True, zero_division=0),
        "test_samples": len(df),
    }

    out = ensure_dir(a.outputs_dir)
    plot_confusion_matrix(confusion_matrix(tc, pc), list(ic.values()), "Disaster CM", out/"cm_disaster.png")
    plot_confusion_matrix(confusion_matrix(ts, ps), list(isev.values()), "Severity CM", out/"cm_severity.png")

    save_json(res, out/"test_metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/best_model.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluate(p.parse_args())