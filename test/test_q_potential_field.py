import numpy as np
import larp

"""
Author: Josue N Rivera

Tests for RiskField's automatic quad decomposition (formerly the separate
QRiskField class -- now just RiskField(..., minimum_cell_size=...)).
"""

def default_min_cell(field):
    return (max(field.size) / 8) * 0.9

def test_eval():

    rgjs = [
        {
            "type": "Point",
            "coordinates": [50, 50],
            "repulsion": [[100, 0], [0, 25]]
        },
        {
            "type": "Point",
            "coordinates": [60, 55],
            "repulsion": [[144, 0], [0, 144]]
        },
        {
            "type": "Point",
            "coordinates": [55, 48],
            "repulsion": [[64, 0], [0, 100]]
        },
        {
            "type": "LineString",
            "coordinates": [[62, 53], [62, 60], [65, 65], [60, 60]],
            "repulsion": [[25, 0], [0, 25]]
        }
    ]

    tmp_field = larp.RiskField(rgjs=rgjs, size=(100, 100))
    qfield = larp.RiskField(rgjs=rgjs, size=(100, 100), minimum_cell_size=default_min_cell(tmp_field))

    x = np.array([[50, 65], [70, 60], [60, 60], [63, 63], [50, 50], [65, 70]])

    out = qfield.eval(x)

    assert len(np.squeeze(out)) == len(x),   "Evaluation of line string rgj does not return a size equal to the size of the input"
    assert np.squeeze(out)[0] != 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[1] != 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[2] == 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[3] == 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[4] == 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[5] == np.exp(-1), "Evaluation of line string rgj is incorrect for a point one magnitude away"

test_eval()

def test_area_estimation():
    rgjs = [
        {
            "type": "Point",
            "coordinates": [50, 50],
            "repulsion": [[1, 0], [0, 1]]
        }
    ]

    tmp_field = larp.RiskField(rgjs=rgjs, size=(100, 100))
    qfield = larp.RiskField(rgjs=rgjs, size=(100, 100), minimum_cell_size=default_min_cell(tmp_field))

    area = qfield.estimate_route_area([(49, 50), (51, 50)], step=0.0001)
    assert ((area - 1.49364)**2).sum() < 1e-5, "Area estimation off"

    area = qfield.estimate_route_area([(49, 50), (51, 50)], step=0.0001, scale_transform=lambda x: 1/(1.0 - x + 0.000001))

test_area_estimation()

def test_gradient():
    # Rectangle/Ellipse are no longer part of the GeoJSON-standard geometry
    # set; the square obstacle below stands in for both (its exact bbox
    # doesn't matter here -- these two are only smoke-tested, not asserted).
    rgjs = [
        {
            "type": "Point",
            "coordinates": [50, 50],
            "repulsion": [[1, 0], [0, 1]]
        },
        {
            "type": "LineString",
            "coordinates": [[10, 10], [10, 20], [20, 20], [20, 10]],
            "repulsion": [[2, 0], [0, 2]]
        },
        {
            "type": "Polygon",
            "coordinates": [[[25, 25], [30, 25], [30, 30], [25, 30], [25, 25]]],
            "repulsion": [[1, 0], [0, 1]]
        },
        {
            "type": "Point",
            "coordinates": [80, 80],
            "repulsion": [[4, 0], [0, 4]]
        }
    ]

    tmp_field = larp.RiskField(rgjs=rgjs, size=(100, 100))
    qfield = larp.RiskField(rgjs=rgjs, size=(100, 100), minimum_cell_size=default_min_cell(tmp_field))

    grad = qfield.gradient([(49, 50), (51, 50), (51, 51)])

    assert ((grad[0] - np.array([2*np.exp(-1), 0]))**2).sum() < 1e-5, "Unexpected gradient"
    assert ((grad[1] - np.array([-2*np.exp(-1), 0]))**2).sum() < 1e-5, "Unexpected gradient"
    assert ((grad[2] - np.array([-2*np.exp(-2), -2*np.exp(-2)]))**2).sum() < 1e-5, "Unexpected gradient"


    grad = qfield.gradient([(11, 10), (20, 11), (32, 31), (81, 82)])

test_gradient()

def test_bbox():
    rgjs = [
        {
            "type": "Point",
            "coordinates": [50, 50],
            "repulsion": [[1, 0], [0, 1]]
        },
        {
            "type": "LineString",
            "coordinates": [[10, 10], [10, 20], [20, 20], [20, 10]],
            "repulsion": [[2, 0], [0, 2]]
        },
        {
            "type": "Polygon",
            "coordinates": [[[25, 25], [30, 25], [30, 30], [25, 30], [25, 25]]],
            "repulsion": [[1, 0], [0, 1]]
        },
        {
            "type": "Polygon",
            "coordinates": [[[78, 78], [82, 78], [82, 82], [78, 82], [78, 78]]],
            "repulsion": [[4, 0], [0, 4]]
        }
    ]

    tmp_field = larp.RiskField(rgjs=rgjs, size=(100, 100))
    field = larp.RiskField(rgjs=rgjs, size=(100, 100))
    qfield = larp.RiskField(rgjs=rgjs, size=(100, 100), minimum_cell_size=default_min_cell(tmp_field))

    assert field.rgjs[1].in_bbox([15, 15]),   "Error determining bbox for linestring"
    assert qfield.find_bbox([81, 80])[0] == 3, "Error finding bbox for polygon"
    assert qfield.find_bbox([15, 15])[0] == 1, "Error finding bbox for linestring"

test_bbox()

def test_add_remove_field():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 60],
        'repulsion': [[5, 0], [0, 5]]
    }]

    field = larp.RiskField(size=50, center_point=[55, 55], rgjs=point_rgjs,
                            minimum_cell_size=5)
    quadtree = field.quadtree

    assert field.eval([(55, 55)])[0] != 1.0, "RiskField eval correct"
    assert quadtree.find_quad([(54.9, 55)])[0].boundary_zone != 0, "Boundary zone correct in quadtree"

    # Add RGJ
    point = larp.PointRGJ((55, 55), repulsion=[[10, 0], [0, 10]])
    added_idx = field.addRGJ(point)

    assert field.eval([(55, 55)])[0] == 1.0, "RGJ not added to risk field"
    assert quadtree.find_quad([(54.9, 55)])[0].boundary_zone == 0, "Boundary zone not update in quadtree"

    # Delete RGJ
    field.delRGJ(added_idx)

    assert field.eval([(55, 55)])[0] != 1.0, "RGJ not removed from risk field"
    assert quadtree.find_quad([(54.9, 55)])[0].boundary_zone != 0, "Boundary zone not update in quadtree"

test_add_remove_field()

def test_add_rgj_idx_passed():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 60],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 50],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [50, 60],
        'repulsion': [[5, 0], [0, 5]]
    }]

    field = larp.RiskField(size=40, center_point=[55, 55], rgjs=point_rgjs,
                            minimum_cell_size=5)
    quadtree = field.quadtree

    field.addRGJ(larp.PointRGJ((55, 55), repulsion=[[25, 0], [0, 25]]))

    def get_rgj_idx(quad:larp.quad.QuadNode):
        assert all(np.array(quad.rgj_idx) < len(field)), f"Quad's rgj indexes {quad.rgj_idx} are out of bound"

        if quad.leaf:
            return

        for child in quad.children:
            assert set(child.rgj_idx) <= set(quad.rgj_idx), f"Child {str(child)} rgj idxs are not a subset of {str(quad)}'s: child = {child.rgj_idx} | parent = {quad.rgj_idx}"
            assert child.boundary_zone >= quad.boundary_zone, f"Child {str(child)} boundary zone is higher than {str(quad)}'s"

        for child in quad.children:
            get_rgj_idx(child)

    get_rgj_idx(quadtree.root)

test_add_rgj_idx_passed()

def test_remove_rgj_idx_passed():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 60],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 50],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [50, 60],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [55, 55],
        'repulsion': [[25, 0], [0, 25]]
    }]

    field = larp.RiskField(size=40, center_point=[55, 55], rgjs=point_rgjs,
                            minimum_cell_size=0.5)
    quadtree = field.quadtree

    field.delRGJ([2, 3])

    def get_rgj_idx(quad:larp.quad.QuadNode):
        assert all(np.array(quad.rgj_idx) < len(field)), f"Quad's rgj indexes {quad.rgj_idx} are out of bound"

        if quad.leaf:
            return

        for child in quad.children:
            assert set(child.rgj_idx) <= set(quad.rgj_idx), f"Child {str(child)} rgj idxs are not a subset of {str(quad)}'s: child = {child.rgj_idx} | parent = {quad.rgj_idx}"
            assert child.boundary_zone >= quad.boundary_zone, f"Child {str(child)} boundary zone is higher than {str(quad)}'s"

            get_rgj_idx(child)

    get_rgj_idx(quadtree.root)

test_remove_rgj_idx_passed()

def test_add_leaf_none_children():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 60],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 50],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [50, 60],
        'repulsion': [[5, 0], [0, 5]]
    }]

    field = larp.RiskField(size=40, center_point=[55, 55], rgjs=point_rgjs,
                            minimum_cell_size=5)
    quadtree = field.quadtree

    field.addRGJ(larp.PointRGJ((55, 55), repulsion=[[25, 0], [0, 25]]))

    for quad in quadtree.leaves:
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by list) with non-none children"

    for quad in quadtree.search_leaves():
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by search) with non-none children"

    assert quadtree.search_leaves() == quadtree.leaves, "Leaves in list are different than those found by search"

test_add_leaf_none_children()

def test_remove_leaf_none_children():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 60],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 50],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [50, 60],
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [55, 55],
        'repulsion': [[25, 0], [0, 25]]
    }]

    field = larp.RiskField(size=40, center_point=[55, 55], rgjs=point_rgjs,
                            minimum_cell_size=1)
    quadtree = field.quadtree

    field.delRGJ([2, 3])

    for quad in quadtree.leaves:
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by list) with non-none children"

    for quad in quadtree.search_leaves():
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by search) with non-none children"

    assert quadtree.search_leaves() == quadtree.leaves, "Leaves in list are different than those found by search"

test_remove_leaf_none_children()
