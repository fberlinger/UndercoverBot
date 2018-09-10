# Ping other fish
PING = b'\x10'
HOMING = b'\x20'

# Send an info to other fish
INFO_EXTERNAL = b'\x31'
INFO_INTERNAL = b'\x32'

# Count message hops
HOP_COUNT = b'\x40'
START_HOP_COUNT = b'\x41'

# Leader election
LEADER_ELECTION = b'\x50'
START_LEADER_ELECTION = b'\x51'

# Let the fish move into a target direction
MOVE = b'\x60'
