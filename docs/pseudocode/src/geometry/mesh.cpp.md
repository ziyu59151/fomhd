# 全域几何、实体与权重

> 状态：active；目标 `src/geometry/mesh.cpp`。依据 GC1/2/5 与全域分册 §2。  
> 依赖：[类型](../../include/fomhd/types.hpp.md)、[输入](../io/config.cpp.md)。同步 operators、wall、NPZ、tests；最近同步：2026-11-03。

## A0-GEO-01 — PolygonAndMask

输入 host `R_nodes[NR],Z_nodes[NZ],polygon[(R,Z)]`，单位 m。输出 `bool active[NZ][NR]`，不修改输入。保存原始输入与判定规则版本 `half_open_ray_v1`。

```text
ValidatePolygonAndGrid(input):
  require finite coordinates; NR,NZ>=2; strictly increasing nodes
  hR=R[1]-R[0]; hZ=Z[1]-Z[0]; require hR,hZ>0
  check each R[i] against R[0]+i*hR, each Z[j] similarly using grid budget
  require R[0]-hR/2>0
  if final polygon vertex == first: use one closure only (retain raw input separately)
  require >=3 vertices; no duplicate vertex, zero-length/repeated segment
  for pairs of non-neighbor segments:
    test oriented segment intersection and collinear overlap; any contact -> BadPolygon
  require nonzero signed shoelace area; record orientation
  # double 输入提升 long double 作判定；近退化且不能可靠分符号时返回 GeometryAmbiguous
  # orientation 的绝对误差界由运算数与 long-double epsilon 计算，不能任意移动轮廓

InsideHalfOpen(p):
  inside=false
  for (a,b) in consecutive polygon segments including last->first:
    if (a.Z<=p.Z<b.Z) or (b.Z<=p.Z<a.Z):
      xcross=a.R+(p.Z-a.Z)*(b.R-a.R)/(b.Z-a.Z)
      if p.R<xcross: inside=not inside
  return inside

for j in [0,NZ),i in [0,NR): active[j,i]=InsideHalfOpen(R[i],Z[j])
require at least one active node
```

点恰在边上也执行相同的严格径向比较和半开 Z 区间，不另给含糊的“稍微向里”。水平边不参与交叉计数。近退化判定使用区间误差界；仅数学上精确的相等进入 tie 规则，其余不确定情况拒绝并给出点/段索引。参考轮廓用于 mask 和几何误差，力/通量一律用下面的阶梯法向。

## A0-GEO-02 — EnumerateAndValidateTopology

```text
cells=sorted active (i,j) by (j,i)
rset,zset,vset=empty ordered sets
for (i,j) in cells:
  insert r(i,j),r(i+1,j); z(i,j),z(i,j+1)
  insert v(i,j),v(i+1,j),v(i,j+1),v(i+1,j+1)
sort each set by (second_index,first_index); assign contiguous IDs
make dense_to_active filled invalid_id; fill inverse maps exactly once

for r(e,j): neighbors=[cell(e-1,j),cell(e,j)] intersect cells
for z(i,f): neighbors=[cell(i,f-1),cell(i,f)] intersect cells
require each has 1 or 2 neighbors
for v(e,f): neighbors=[(e-1,f-1),(e,f-1),(e-1,f),(e,f)] intersect cells
  boundary_v=(count<4)
  reject exactly two diagonal neighbors without an edge-connected pair
for each r/z with one neighbor c:
  mark wall; normal points from c center to face center
  r: sigma=+1 if e=i_c+1 else -1; normal=(sigma,0,0)
  z: sigma=+1 if f=j_c+1 else -1; normal=(0,0,sigma)
  store two endpoints,face_id,cell_near,cell_next along -normal
  require cell_next exists, adjacent and active

BFS active cells via shared faces; require one connected component
wall_graph: vertices=wall segment endpoints, edges=wall segments
require every wall vertex degree=2; BFS wall_graph require one component
require Nv-(Nr+Nz)+Nc==1
reject any self-contact or nonmanifold topology
```

以上是单个极向圆盘；加上周期环向后允许原有磁拓扑通量，但不增加环路电压。壁支撑不足不是“改用一阶”的开关。错误回报 wall ID、两个应有的节点键和轮廓哈希。

## A0-GEO-03 — Metrics

```text
for c(i,j): Rc=R0+i*hR; Zc=Z0+j*hZ; Vc=Rc*hR*hZ
for r(e,j):
  Rr=R0+(e-0.5)*hR; Zr=Z0+j*hZ
  lr=hR if internal else hR/2
  Rbar_r=Rr if internal else (Rr+Rc_neighbor)/2
for z(i,f):
  Rz=R0+i*hR; Zz=Z0+(f-0.5)*hZ
  lz=hZ if internal else hZ/2
for v(e,f): av=0; wv=0
for c(i,j),corner(e,f) in its four corners:
  Rhalf=R0+(e-0.5)*hR
  Rlo=min(Rc,Rhalf); Rhi=max(Rc,Rhalf)
  av[v]+=hR*hZ/4
  wv[v]+=(Rhi*Rhi-Rlo*Rlo)*hZ/4
for v: Rbar_v=wv/av
M0=Vc
M1=concat(Rr*hZ*lr, Vc, Rz*hR*lz)
MB=concat(Rz*hR*lz, wv, Rbar_r*hZ*lr)
for wall r: area_per_radian=Rr*hZ
for wall z: area_per_radian=Rz*hR
require every R,weight,length,area positive and finite
```

权按每单位环向角计算，积分由诊断乘 `2*pi` 或 `2*pi/Nphi`；不能重复乘 R。M1/MB 壁条目是本离散的 Hodge 权，不全部等于曲面控制体精确体积。两个径向 R 不可合并。

## A0-GEO-04 — GeometryDistanceAndHash

```text
PointSegmentDistance(p,a,b):
  t=clamp(dot(p-a,b-a)/dot(b-a,b-a),0,1)
  return norm(p-(a+t*(b-a)))
eta=min(hR,hZ)/8
d=0
for (source,target) in [(polygon,wall_polyline),(wall_polyline,polygon)]:
  for segment (a,b) of source:
    q=ceil(length(b-a)/eta)
    for s in [0,q] inclusive:
      p=a+(s/q)*(b-a)
      distance=min(PointSegmentDistance(p,ta,tb) for each target segment)
      d=max(d,distance)
hausdorff_interval=[d,d+eta/2]
for entity: distance_to_wall=min exact segment distance at its point
region_inner=(distance_to_wall>=config.diagnostics.inner_distance_m)
region_wall=not region_inner
geometry_hash=SHA256(canonical little-endian serialization of grid,polygon,
  active mask,entity keys,wall normals,metric arrays,rule_version)
```

源为解析曲线时 provider 另给折线逼近误差；这里的上界仅针对两条已给折线。内域误差区使用固定物理距离，不能随 h 改成恒定两格而虚报四阶。哈希内的整数宽度、字节序和数组顺序由 NPZ/manifest schema 固定；浮点 -0 规范为 +0，禁止 NaN。

后置检查：拓扑数量、wall 支撑、正权、几何距离及矩阵内存计数写初始化诊断；几何失败不进入线性分解。对应测试 `GEO-01..04`。
