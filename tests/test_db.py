from recommender.data.db import get_engine, create_tables, get_session, Image


def test_insert_and_retrieve_image(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    create_tables(engine)
    session = get_session(engine)

    img = Image(path="test.jpg", width=100, height=200)
    session.add(img)
    session.commit()

    result = session.query(Image).first()

    assert result.path == "test.jpg"
    assert result.width == 100
    assert result.height == 200
