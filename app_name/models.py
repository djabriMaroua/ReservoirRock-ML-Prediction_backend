from django.db import models

from django.db import models

class LasData(models.Model):
    depth = models.FloatField()
    nphi = models.FloatField()
    rhob = models.FloatField()
    gr = models.FloatField()
    rt = models.FloatField()
    pef = models.FloatField()
    cali = models.FloatField()
    dt = models.FloatField()
