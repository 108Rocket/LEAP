"""LEAP input interface; the TrajAgg encoder is in upstream/."""


def construct(metric):
    from external_models.reproduction.sdd_porto.sdd_models import construct as create
    return create('TrajAgg', metric)
