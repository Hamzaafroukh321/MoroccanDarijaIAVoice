"""Select task adapters from explicit contracts, shared by live sessions/recovery."""


def make_task(config):
    if not config.get('demo'):
        from engine.state import TaskState
        return TaskState(config)
    settings = config['demo']
    kind = settings.get('state_kind')
    if kind == 'configured_collection_scoped':
        from engine.collection_task import ConfiguredCollectionState
        return ConfiguredCollectionState(config)
    if kind == 'flat_scoped':
        from engine.scoped_task import ScopedTaskState
        return ScopedTaskState(config)
    if kind not in {None, 'collection_scoped'}:
        raise ValueError('Unknown task adapter kind.')
    if settings.get('order_schema_version') == 2:
        from engine.demo_order import DemoOrderState
        return DemoOrderState(config)
    # The explicit legacy single-order preview remains supported.
    from engine.demo import DemoTaskState
    return DemoTaskState(config)
