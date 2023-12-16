# coding=utf-8
from pydantic import BaseModel

from app.utils.system_stats import CloudSystemStats


class IdentifyResult(BaseModel):
    id: str
    uid: str
    name: str
    time: str
    score: float


class SystemStats(BaseModel):
    cpu_percent: float
    ram_percent: float
    net_throughput: float

    def __init__(self, cloud_system_stats: CloudSystemStats):
        super().__init__(
            cpu_percent=cloud_system_stats.get_cpu_usage(),
            ram_percent=cloud_system_stats.get_ram_usage(),
            net_throughput=cloud_system_stats.get_network_throughput()
        )
