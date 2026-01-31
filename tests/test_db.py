from imagerec.db import ImageDB


def test_db_insert_and_fetch(tmp_path):
    db_path = tmp_path / "test.db"
    db = ImageDB(str(db_path))
    db.init_schema()

    rows = [("/tmp/img1.jpg", 100, 200, "JPEG"), ("/tmp/img2.jpg", 64, 64, "JPEG")]
    mapping = db.insert_images(rows)
    assert "/tmp/img1.jpg" in mapping
    assert "/tmp/img2.jpg" in mapping

    image_id = mapping["/tmp/img1.jpg"]
    fetched = db.fetch_image(image_id)
    assert fetched[1] == "/tmp/img1.jpg"

    db.close()
