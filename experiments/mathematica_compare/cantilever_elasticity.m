Needs["NDSolve`FEM`"]
{objPath, femDegree, Young, poisson} = {$ScriptCommandLine[[2]], ToExpression[$ScriptCommandLine[[3]]], ToExpression[$ScriptCommandLine[[4]]], ToExpression[$ScriptCommandLine[[5]]]};
objMesh = Import[objPath];
(* Annoyingly, I cannot seem to prevent ToElementMesh from remeshing when passed objMesh directly. It seems I must manually convert it. *)
pts = MeshCoordinates[objMesh];
tris = Map[#1[[1]] &, MeshCells[objMesh, 2]]; (* strip off Polygon[] *)
em = ToElementMesh["Coordinates"->pts, "MeshElements"->{TriangleElement[tris]}];
em = MeshOrderAlteration[em, femDegree];
linearElasticity[Y_,nu_] := {Inactive[Div][({{0,-((Y nu)/(1-nu^2))},{-((Y (1-nu))/(2 (1-nu^2))),0}}.Inactive[Grad][v[x,y],{x,y}]),{x,y}]+Inactive[Div][({{-(Y/(1-nu^2)),0},{0,-((Y (1-nu))/(2 (1-nu^2)))}}.Inactive[Grad][u[x,y],{x,y}]),{x,y}],Inactive[Div][({{0,-((Y (1-nu))/(2 (1-nu^2)))},{-((Y nu)/(1-nu^2)),0}}.Inactive[Grad][u[x,y],{x,y}]),{x,y}]+Inactive[Div][({{-((Y (1-nu))/(2 (1-nu^2))),0},{0,-(Y/(1-nu^2))}}.Inactive[Grad][v[x,y],{x,y}]),{x,y}]};
dirichletBC = DirichletCondition[{u[x,y]==0,v[x,y]==0.},x==-1];
traction = NeumannValue[-10, x==1];
{qu,qv} = NDSolveValue[{linearElasticity[Young, poisson]=={0,traction}, dirichletBC},{u,v},{x,y} \[Element]em];
(* eval[x_,y_] := Map[NumberForm[#1[x, y], 16] &, {qu, qv}]; *)
eval[x_,y_] := Map[CForm[#1[x, y]] &, {qu, qv}];
printSample[x_,y_] := Print[StringJoin[Insert[ToString /@ eval[x, y], " ", 2]]]
printSample[1,1]
printSample[0.5,0.5]
printSample[0.38,0.38]
