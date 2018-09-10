from eventcodes import (
    INFO_EXTERNAL, INFO_INTERNAL, PING, HOMING, HOP_COUNT, START_HOP_COUNT,
    MOVE, LEADER_ELECTION, START_LEADER_ELECTION
)


class Ping():
    """Ping your beloved neighbor fish
    """

    def __init__(self, id):
        """Create a ping event to "sense" other fish

        Arguments:
            id {int} -- ID of the fish that spreads this event.
        """

        self.opcode = PING
        self.source_id = id


class Homing():
    """Homing towards an external source
    """

    def __init__(self):
        """Create new homing event.

        In robotics homing is equivalent to aggregation. A.k.a.: "Dear fellow
        fish friends, let's come home to mommy and daddy and enjoy a nice cup
        of chocolate."
        """

        self.opcode = HOMING


class Move():
    """Make the fish move to a target direction
    """

    def __init__(self, x=0, y=0):
        """External event to make fish start moving into a target direction

        Keyword Arguments:
            x {number} -- X target position (default: {0})
            y {number} -- Y target position (default: {0})
        """

        self.opcode = MOVE
        self.x = x
        self.y = y


class HopCount():
    """Broadcast hop counts

    A funny side note: in Germany distributed and DNA-based organisms (often
    called humans) shout "Hop Hop rin in Kopp", which is a similar but slightly
    different event type that makes other human instances to instantly enjoy
    a whole glass of juicy beer in just a single hop! Highly efficient!
    """

    def __init__(self, id, clock, hops=0):
        """Create an internal information event

        Arguments:
            id {int} -- ID of the fish that spreads this event.
            clock {int} -- Clock value at the time the information was first
                observed.

        Keyword Arguments:
            hops {int} -- Number of hops this information is already jumping
                around through our beloved fish swarm (default: {0})
        """

        self.opcode = HOP_COUNT
        self.source_id = id
        self.clock = clock
        self.hops = hops


class StartHopCount():
    """Initialize a hop count.
    """

    def __init__(self):
        """External event to make fish start a hop count
        """

        self.opcode = START_HOP_COUNT


class LeaderElection():
    """Broadcast a leader election
    """

    def __init__(self, id, max_id):
        """Create an internal leader lection event

        Arguments:
            id {int} -- ID of the fish that spreads this event.
            max_id {int} -- Maximum fish ID, which will be the final leader.
        """

        self.opcode = LEADER_ELECTION
        self.source_id = id
        self.max_id = max_id


class StartLeaderElection():
    """Initialize a leader election
    """

    def __init__(self):
        """External event to make fish start a leader election
        """

        self.opcode = START_LEADER_ELECTION


class InfoInternal():
    """Share information internally with other fish
    """

    def __init__(self, id, clock, message, hops=0):
        """Create an internal information event

        Arguments:
            id {int} -- ID of the fish that spreads this event.
            clock {int} -- Clock value at the time the information was first
                observed.
            message {*} -- Some information. In most cases this is just a
                number.

        Keyword Arguments:
            hops {int} -- Number of hops this information is already jumping
                around through our beloved fish swarm (default: {0})
        """

        self.opcode = INFO_INTERNAL
        self.source_id = id
        self.clock = clock
        self.message = message
        self.hops = hops


class InfoExternal():
    """Share external information with fish
    """

    def __init__(self, message, track=False):
        """Create an external information event

        Arguments:
            message {*} -- Some information. In most cases this is just a
                number.

        Keyword Arguments:
            track {bool} -- If `true` the event will be tracked by the observer
                (default: {False})
        """

        self.opcode = INFO_EXTERNAL
        self.message = message
        self.track = track
