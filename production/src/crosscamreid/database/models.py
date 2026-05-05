from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, JSON, String, Text
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class ReidDetection(Base):
    __tablename__ = "reid_detections"

    id     = Column(Integer, primary_key=True, index=True, autoincrement=True)
    camid  = Column(String(100), nullable=True)
    org_id = Column(Integer, nullable=True)
    userid = Column(Integer, nullable=True)

    frame_id   = Column(String, unique=True, index=True)
    time_stamp = Column(DateTime(timezone=True), nullable=True)
    frame_count = Column(Integer, nullable=True)
    fps         = Column(Float, nullable=True)

    total_detected   = Column(Integer, nullable=True)
    locked_count     = Column(Integer, nullable=True)   # persons in LOCK phase
    qualifying_count = Column(Integer, nullable=True)   # persons in QUALIFY/ENROLL

    # Per-person arrays — one entry per detected person, ordered consistently
    track_ids        = Column(JSON, nullable=True)   # list[int]
    sids             = Column(JSON, nullable=True)   # list[str]   subject IDs
    phases           = Column(JSON, nullable=True)   # list[str]   QUALIFY/ENROLL/LOCK/NEW
    bounding_boxes   = Column(JSON, nullable=True)   # list[[x1,y1,x2,y2]]
    similarity_scores = Column(JSON, nullable=True)  # list[float|null]
    keypoint_valid   = Column(JSON, nullable=True)   # list[bool]

    status             = Column(Text, nullable=True)
    is_alert_triggered = Column(Boolean, default=False)
    processing_status  = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ReidStreamFrame(Base):
    __tablename__ = "reid_stream_frames"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    stream_frame_id = Column(String, unique=True, index=True, nullable=False)
    client_id = Column(String(128), nullable=True, index=True)
    org_id = Column(Integer, nullable=True)
    userid = Column(Integer, nullable=True)

    requested_cameras = Column(JSON, nullable=False)   # list[str]
    camera_payloads = Column(JSON, nullable=False)     # {"cam1": {...}, "cam2": {...}}

    time_stamp = Column(DateTime(timezone=True), nullable=True)
    fps = Column(Float, nullable=True)
    camera_count = Column(Integer, nullable=True)
    total_detected = Column(Integer, nullable=True)
    status = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
