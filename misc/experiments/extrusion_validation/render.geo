General.SmallAxes = 0;

General.GraphicsWidth = 1024;
General.GraphicsHeight = 768;

General.RotationX = 10.49430951496418;
General.RotationY = 28.15524358106397;
General.RotationZ = 4.043063394167416;
General.RotationCenterGravity = 1;
General.RotationCenterX = 0;
General.RotationCenterY = 0;
General.RotationCenterZ = 0;
General.Trackball = 1;
General.TrackballHyperbolicSheet = 1;
General.TrackballQuaternion0 = -0.09719460333923287;
General.TrackballQuaternion1 = -0.2389370571657026;
General.TrackballQuaternion2 = -0.05630290239750187;
General.TrackballQuaternion3 = 0.9645166017111048;
General.TranslationX = 0;
General.TranslationY = 0;
General.TranslationZ = 0;
General.Orthographic = 0;
General.Orthographic = 0;


// Vector field: displacement
View[v].VectorType = 5;
View[0].Visible = 1;
Draw;
Print "u_perspective.png";

General.TrackballQuaternion0 = 0;
General.TrackballQuaternion1 = 0;
General.TrackballQuaternion2 = 0;
General.TrackballQuaternion3 = 1;
General.Orthographic = 1;
Draw;
Print "u_top.png";

Exit;
