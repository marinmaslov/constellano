// Name: Grep negatives
// License: MIT License
// Author: marinmaslov
// Last Modified: 2022/04/16
// Description: Take snaps of northen sky except for defined FOV

// EQ COORDINATES
// RA coordinates (decimal) [up-down]
const RA_TARGET_START = 279.42;
const RA_TARGET_END = 284.94;

// DEC coordinates (decimal) [up-down]
const DEC_TARGET_START = 32.68505556;
const DEC_TARGET_END = 38.79933333;


LabelMgr.labelScreen("Taking negative shoots'.", 200, 200, true, 20, "#ff0000");
core.wait(2);
LabelMgr.deleteAllLabels();
StelMovementMgr.zoomTo(18.0, 1);
StelSkyDrawer.setFlagStarMagnitudeLimit(true);
StelSkyDrawer.setCustomStarMagnitudeLimit(6.5);
core.setGuiVisible(false);
core.setProjectionMode("ProjectionPerspective");
core.wait(3);

//Movement code
LabelMgr.deleteAllLabels();
LabelMgr.labelScreen("Capturing sample images.", 200, 200, true, 20, "#ff0000");
core.wait(2);
LabelMgr.deleteAllLabels();
core.wait(2);


var targetRadiusCheck = StelMovementMgr.getCurrentFov() / 2;

for (var i = 0.0; i <=90.0; i = i + 5.0) {
	for (var j = 0.0; j <= 360.0; j = j + 5.0) {
		if (!(i + targetRadiusCheck < RA_TARGET_START && i - targetRadiusCheck < RA_TARGET_END && j + targetRadiusCheck < DEC_TARGET_START && j - targetRadiusCheck < DEC_TARGET_END)) {
			core.moveToRaDec(j, i, 0);
			core.screenshot("negative_", false, "C:/Users/easmsma/Desktop/Diplomski/constellation-recognition/constellation-recognition/targets/lyra/negative/", false, "jpg");
			core.wait(0.5);
		}
		core.wait(0.5);
	}
	core.wait(1);
}
core.wait(2);
core.setGuiVisible(true);