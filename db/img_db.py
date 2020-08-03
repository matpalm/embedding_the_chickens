import psycopg2
from detections.detection import Detection
from file_util import split_fname


class ImgDB(object):
    def __init__(self):
        self.conn = psycopg2.connect("dbname=etc user=postgres")

    def insert_fname_entries(self, fnames):
        values = []
        for fname in fnames:
            cam, ymd, hms = split_fname(fname)
            values.append((cam, f"{ymd} {hms}", fname))
        c = self.conn.cursor()
        c.executemany(
            "insert into imgs (cam, dts, fname) values (%s, %s, %s)", values)
        self.conn.commit()

    def fnames_without_detections(self):
        c = self.conn.cursor()
        c.execute("select id, fname from imgs where detections_run is false")
        return c.fetchall()

    def set_detections(self, img_id, detections):
        c = self.conn.cursor()
        values = [(img_id, d.entity, d.score, d.x0, d.y0, d.x1, d.y1)
                  for d in detections]
        c.executemany("insert into detections (img_id, entity, score, x0, y0, x1, y1)"
                      " values (%s,%s,%s,%s,%s,%s,%s)", values)
        c.execute("update imgs set detections_run=true where id=%s" %
                  (img_id,))
        self.conn.commit()

    def detections_for_img(self, fname):
        c = self.conn.cursor()
        c.execute("select d.id, d.entity, d.score, d.x0, d.y0, d.x1, d.y1"
                  " from detections d join imgs i on d.img_id=i.id"
                  " where i.fname='%s' order by d.id" % (fname,))
        return list(map(Detection._make, c.fetchall()))
