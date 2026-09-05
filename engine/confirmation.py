"""Shared acceptance guard for the exact task version whose readback completed."""


class VersionedConfirmation:
    """Task adapters supply readiness and any unresolved clarification blocker."""

    @property
    def confirmation_blocked(self):
        return False

    @property
    def confirmed(self):
        return (self.ready and not self.confirmation_blocked and
                not self.awaiting_correction and self.confirmed_version == self.version)

    def invalidate_confirmation(self):
        self.readback_version = None
        self.confirmed_version = None

    def begin_confirmation(self, version):
        # Playback acknowledgements cannot approve older task versions.
        if type(version) is int and self.ready and version == self.version and not self.confirmation_blocked:
            self.readback_version = version
            self.awaiting_correction = False

    def accept_confirmation(self, operations):
        # Even a redundant correction plus "yes" requires a fresh readback.
        if operations:
            self.invalidate_confirmation()
            return False
        if (self.ready and not self.confirmation_blocked and not self.awaiting_correction and
                self.readback_version == self.version):
            self.confirmed_version = self.version
            return True
        return False
