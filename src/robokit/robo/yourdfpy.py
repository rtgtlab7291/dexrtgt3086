"""
This is a wrapper of yourdfpy.URDF that adds mimic joint writing.

Consider submitting a PR to yourdfpy to add this feature in the future.
"""

from lxml import etree
from yourdfpy import URDF


class URDFWrapper(URDF):
    def _write_joint(self, xml_parent, joint):
        xml_element = etree.SubElement(
            xml_parent,
            "joint",
            attrib={
                "name": joint.name,
                "type": joint.type,
            },
        )

        etree.SubElement(xml_element, "parent", attrib={"link": joint.parent})
        etree.SubElement(xml_element, "child", attrib={"link": joint.child})
        self._write_origin(xml_element, joint.origin)
        self._write_axis(xml_element, joint.axis)
        self._write_limit(xml_element, joint.limit)
        self._write_dynamics(xml_element, joint.dynamics)

        # NOTE: different from the original yourdfpy, we add mimic joint writing here
        if joint.mimic is not None:
            self._write_mimic(xml_element, joint.mimic)
