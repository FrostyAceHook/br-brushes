import numpy as np
from collections import OrderedDict
from pymclevel import alphaMaterials, BoundingBox


displayName = "Spike"
disableStyleButton = True


def createInputs(self):
    self.inputs = (
        {"Block": alphaMaterials.Stone},
        {"Replace?": True},
        {"Replace:": alphaMaterials.Air},
        {"Length:": (10.0, 0.1, 256.0)},
        {"Radius:": (5.0, 0.0, 256.0)},
        {"Hollow radius:": (0.0, 0.0, 256.0)},
        {"": "Direction:"}, # weird setup to do a label.
        OrderedDict([("Inwards:", True), ("Outwards:", True)]),
        {"Minimum Spacing": (1, 0, 100)},
    )


# Box centred around point which is the maximum the spike can reach. Note this
# doesn't take the spike direction into account, because the player position can
# change without re-calling `createDirtyBox`.
def max_bounds(point, length, rad_max):
    outer = max(length, rad_max) + 1
    origin = tuple(c - outer for c in point)
    size = (2*outer + 1,) * 3
    return BoundingBox(origin, size)


def createDirtyBox(self, point, tool):
    return max_bounds(point, tool.options["Length:"], tool.options["Radius:"])


def apply(self, op, point):
    options = op.options
    editor = op.editor
    level = op.level

    replace = options["Replace?"]
    rid, rdata = options["Replace:"].ID, options["Replace:"].blockData
    bid, bdata = options["Block"].ID, options["Block"].blockData

    length = options["Length:"]
    rad_max, rad_min = options["Radius:"], options["Hollow radius:"]
    inwards, outwards = options["Inwards:"], options["Outwards:"]


    # Get the maximum possible spike box (accounts for any direction).
    box = max_bounds(point, length, rad_max)

    # Spike starting point is at the cursor, "ending" point is at the camera.
    # Note the end point is only used for direction, not distance.
    start = point
    end = editor.mainViewport.cameraPosition


    # If `rad_min > rad_max`, the entire spike would be hollow. It maybe should
    # be an exception, but exceptions in brushes are handled kinda poorly in
    # mcedit (e.g., the brush points persist despite not clicking), so we just
    # silently ignore it.
    if rad_min > rad_max:
        return

    # Same kinda idea if neither direction is selected.
    if not inwards and not outwards:
        return


    # Get the normalised spike direction.
    dir_x = end[0] - start[0]
    dir_y = end[1] - start[1]
    dir_z = end[2] - start[2]
    dir_mag = np.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
    dir_x /= dir_mag
    dir_y /= dir_mag
    dir_z /= dir_mag

    # Precompute the inverse so we can multiply.
    neg_inv_length = -1.0 / length


    # Iterate through all the blocks the spike could reach.
    for chunk, slices, point in iterate(level, box):
        ids = chunk.Blocks[slices]
        datas = chunk.Data[slices]


        # Spike algorithm courtesy of 14er, from 14eredit:
        # https://github.com/14ercooper/14erEdit
        # (currently under the spike block iterator at:
        #  src/main/java/com/_14ercooper/worldeditor/blockiterator/iterators/SpikeIterator.java)

        x_offset = box.minx + point[0] - start[0]
        y_offset = box.miny + point[1] - start[1]
        z_offset = box.minz + point[2] - start[2]

        def dist_along_f(x, z, y):
            x += x_offset
            y += y_offset
            z += z_offset

            return (x * dir_x) + (y * dir_y) + (z * dir_z)

        def dist_to_f(x, z, y):
            x += x_offset
            y += y_offset
            z += z_offset

            d = (x * dir_x) + (y * dir_y) + (z * dir_z)
            return np.sqrt(x*x + y*y + z*z - d*d)

        dist_along = np.fromfunction(dist_along_f, ids.shape, dtype=float)
        dist_to = np.fromfunction(dist_to_f, ids.shape, dtype=float)

        tmp = np.absolute(dist_along) * neg_inv_length + 1.0

        # Get the mask for the double-ended solid spike.
        rad_outer = tmp*rad_max
        mask = (dist_to <= rad_outer)

        # Hollow it out, if requested (since `rad_min` is often zero, we skip
        # this if we can).
        if rad_min > 0.0:
            rad_inner = tmp*rad_min
            mask &= (dist_to >= rad_inner)

        # The spike is currently double-ended, so cull it to the requested
        # directions.
        if not inwards:
            mask &= (dist_along <= 0.0)
        if not outwards:
            mask &= (dist_along >= 0.0)

        # Only modify the matching blocks, if requested.
        if replace:
            mask &= ((ids == rid) & (datas == rdata))


        ids[mask] = bid
        datas[mask] = bdata

    return


# Yoinked from br-filters `br.py::iterate`, with `method=DEFAULT`.
def iterate(level, box):
    # Keep the logic that attempts to prevent the infamous chunk-skipping bug.
    chunks = []
    for chunk, slices, point in level.getChunkSlices(box):
        chunk.chunkChanged(calcLighting=True)
        chunks.append(chunk)

    # Forward from the normal iterator.
    for chunk, slices, point in level.getChunkSlices(box):
        yield chunk, slices, point
