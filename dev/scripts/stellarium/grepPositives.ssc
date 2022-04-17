// Name: Grep Wanted Alt/Az Field Images
// License: MIT License
// Author: marinmaslov
// Last Modified: 2022/04/16
// Description: Use Alt/Az coordinates to take screenshots of ROI


// ALT coordinates should be in decimal degrees (up-down)
const ALT_BORDER_START = 20.0;
const ALT_BORDER_END = 30.0;

// AZ coordinates should be in decimal degrees (left-right)
const AZ_BORDER_START = 55.0;
const AZ_BORDER_END = 70.0;



// EQ COORDINATES
// RA coordinates (decimal) [up-down]
const RA_TARGET_START = 279.42;
const RA_TARGET_END = 284.94;

// DEC coordinates (decimal) [up-down]
const DEC_TARGET_START = 32.68505556;
const DEC_TARGET_END = 38.79933333;


LabelMgr.labelScreen("Taking shoots'.", 200, 200, true, 20, "#ff0000");
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

for (var i = DEC_TARGET_START; i < DEC_TARGET_END; i=i+0.5) {
	for (var j = RA_TARGET_START; j < RA_TARGET_END; j=j+0.5) {
		core.moveToRaDec(i, j, 0);
		core.screenshot("target_", false, "C:/Users/easmsma/Desktop/Diplomski/constellation-recognition/constellation-recognition/targets/lyra", false, "jpg");
		core.wait(0.5);
	}
	core.wait(1);
}
core.wait(2);
core.setGuiVisible(true);