from pathlib import Path
from scripts.ingest_dataset import load_csv_dataset, load_json_or_jsonl


def test_load_csv_dataset(tmp_path: Path) -> None:
    csv_file = tmp_path / "faqs.csv"
    csv_file.write_text(
        'question,answer,category\n'
        '"How to reboot?","Unplug power cable for 10 seconds.","General"\n'
        '"How to reboot?","Unplug power cable for 10 seconds.","General"\n'
        '"What is LED blue?","Blue indicates connected status.","LED Status"\n',
        encoding="utf-8",
    )

    chunks = load_csv_dataset(csv_file, "Test Product", "test-product", hardware_version="V1")
    # Duplicate record should be removed
    assert len(chunks) == 2
    assert "Unplug power cable" in chunks[0].text
    assert chunks[0].metadata["product_id"] == "test-product"
    assert chunks[0].metadata["hardware_version"] == "V1"
    assert chunks[0].metadata["document_type"] == "faq"


def test_load_json_dataset(tmp_path: Path) -> None:
    json_file = tmp_path / "manual.json"
    json_file.write_text(
        '[\n'
        '  {"question": "How to turn on?", "answer": "Press power button."},\n'
        '  {"question": "How to pair?", "answer": "Hold Bluetooth button 5 seconds."}\n'
        ']',
        encoding="utf-8",
    )

    chunks = load_json_or_jsonl(json_file, "Smart Speaker", "smart-speaker")
    assert len(chunks) == 2
    assert "Press power button." in chunks[0].text
    assert chunks[0].metadata["product_id"] == "smart-speaker"
    assert chunks[0].metadata["source_type"] == "json"


def test_load_jsonl_dataset(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "records.jsonl"
    jsonl_file.write_text(
        '{"question": "Reset device", "answer": "Press reset pinhole.", "hardware_version": "V2"}\n'
        '{"question": "Check battery", "answer": "Press battery indicator button.", "hardware_version": "V2"}\n',
        encoding="utf-8",
    )

    chunks = load_json_or_jsonl(jsonl_file, "Sensor Node", "sensor-node")
    assert len(chunks) == 2
    assert chunks[0].metadata["hardware_version"] == "V2"
    assert chunks[0].metadata["source_type"] == "jsonl"
