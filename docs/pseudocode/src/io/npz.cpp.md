# NPZ/NPY 原生数组格式

> 状态：active；目标 `src/io/npz.cpp`。依据 A0 §9.1；首个实际文件写出后破坏性改变须提升 schema。  
> 依赖：[类型](../../include/fomhd/types.hpp.md)、[config](config.cpp.md)。同步 restart、provider 与往返 tests；最近同步：2026-11-03。

## A0-NPZ-01 — LogicalArrays

磁场数组存点值 T，不存磁通或矢势。磁场与流体的矩形磁盘布局如下；紧凑 active 布局只用于运行内存。非活跃位置规范写 complex zero，由几何映射忽略；非零非活跃输入拒绝，禁止 NaN 哨兵混进物理积分。

| 键 | dtype / C-order shape |
|---|---|
| `n_i,p_i,n_e,p_perp_e,p_parallel_e` | `<c16 [M,NZ,NR]` |
| `mom_i,mom_e` | `<c16 [3,M,NZ,NR]`，component R/phi/Z |
| `B_R` | `<c16 [M,NZ+1,NR]`，z 实体 |
| `B_phi` | `<c16 [M,NZ+1,NR+1]`，v 实体 |
| `B_Z` | `<c16 [M,NZ,NR+1]`，r 实体 |
| `R_nodes,Z_nodes,lcfs_R,lcfs_Z` | `<f8 [NR]/[NZ]/[Npolygon]/[Npolygon]` |
| `step,layout_version,schema_version` | `<u8` scalar shape `()` |
| `time,dt,start_time` | `<f8` scalar |
| `model_id,config_hash,semantic_hash,geometry_hash,operator_rule_hash,matrix_fingerprint,original_run_id` | `<Ulength` scalar；UTF-32 code points，禁止 object |
| `accepted` | `|b1` scalar（0/1） |
| `cumulative_json` | `<Ulength` scalar；有限数值 JSON，包含 typed ledger 与 last interval ID |
| `baseline_divB`（checkpoint） | `<c16 [M,NZ+1,NR+1]`，v 实体的原初始散度，T/m；其余原基线元数据在 cumulative_json |

initial 只含主规格固定的十个演化键，units/layout/provider metadata 由 case+source_manifest 绑定；snapshot/checkpoint 在这些数组上增加上述必需元数据。geometry 不接受外部预先计算的 mask 覆盖本地枚举。可选快照派生键用 `derived_` 前缀并在 manifest 描述，restart 从不消费派生字段。

## A0-NPZ-02 — ReadNPZ

ZIP 采用 miniz 3.0.2 成熟库；NPY 仅实现明确的有限 dtype/header 子集。此选择不引入 Python 运行依赖。整个 archive 最大字节、条目数、逐项解压大小和总解压大小都先检查；不执行 pickle/eval，也不把条目解压到任意文件路径。

```text
ReadNPZ(file,expected_schema,limits):
  require regular file and file_size<=max_input_bytes
  archive=zeroed mz_zip_archive
  require mz_zip_reader_init_file(&archive,file,flags=0)
  n=mz_zip_reader_get_num_files; require n<=max_entries
  for idx in [0,n):
    stat=mz_zip_reader_file_stat(archive,idx); require success
    name=full filename via mz_zip_reader_get_filename; bounded length
    require name matches exact allowed key+'.npy',not directory,no duplicate
    reject encrypted entries,unsupported compression,unexpected key
    checked sum of stat.m_uncomp_size; require <=max_uncompressed_bytes
    require stat.m_uncomp_size<=SIZE_MAX and per_array expected cap
    allocate owning bytes of exact declared size
    require mz_zip_reader_extract_to_mem(archive,idx,bytes,size,flags=0)
    # 库负责 ZIP/CRC；失败不是空数组成功
    array=ParseNPY(bytes,expected dtype/shape,limits)
  require all required keys exist exactly once
  reader_end on all paths via RAII
  return arrays
```

读取支持 ZIP stored/deflated；写入固定无压缩。库返回信息不足以检查时拒绝，不自行跳过 CRC。ZIP64 的 size 使用 64 位 checked 转换；压缩体积小不能免除展开预算。

## A0-NPZ-03 — ParseNPY / EncodeNPY

```text
ParseNPY(bytes,schema):
  require bytes.size>=8 before reading magic/version
  require first 6 bytes == 0x93+'NUMPY'; version=(major,minor)
  allow (1,0),(2,0),(3,0)
  require bytes.size>=10 if v1 else bytes.size>=12 before reading header_len
  header_len=read little-endian u16(v1) or u32(v2/v3)
  require header_len<=max_header_bytes and prefix+header_len<=bytes.size
  decode header ASCII(v1/2) or UTF8(v3)
  parse restricted dictionary grammar with exactly keys descr,fortran_order,shape:
    tokens={ },:,(,), single/double quoted ASCII strings, True/False,
      unsigned decimal integer and whitespace; optional trailing commas
    reject any extra expression,escape outside accepted quoted strings,
      duplicate key,negative dimension,unknown key or nested object
  require fortran_order==False
  descr: parse endian '<','>','|','=' and exact scalar kind f8,c16,u8,b1,U<count>
  require scalar kind/width and shape match schema; accept explicit < or > byte order
  reject object/structured dtype and = for multi-byte values; | only allowed for b1
  count=checked product(shape),scalar shape () means count=1
  itemsize f8/u8=8,c16=16,b1=1,Ucount=4*count_chars
  require payload_length==count*itemsize (no trailing/truncated payload)
  for each item: decode bytes explicitly with endian conversion
    complex: decode two binary64 real/imag; do not reinterpret arbitrary C++ struct
    Unicode: decode 32-bit code points, reject invalid scalar/surrogate;
      trailing NUL padding allowed, embedded NUL rejected for metadata strings
  return typed owned array

EncodeNPY(array):
  describe canonical little-endian dtype and C-order shape
  header="{'descr': '<c16', 'fortran_order': False, 'shape': (...), }" (actual dtype)
  append spaces and final newline so prefix+header length is multiple of 64
  choose v1 if length fits u16 else v2 if fits u32 and header cap
  append each explicitly encoded little-endian value
  Unicode length=max(1,number_of_codepoints),zero pad unused slots
  return bytes
```

`=` 字节序需记录生成端信息才可接受；可移植输入要求显式 `<` 或 `>`，首版拒绝 `=` 多字节 dtype。`|` 仅用于 b1。f8/c16 中 NaN/Inf 可在失败文件保留原 bits；普通初态/重启由 state 校验拒绝。

## A0-NPZ-04 — WriteNPZAndRoundTrip

```text
WriteNPZToNewTempFile(temp_path,arrays):
  host_arrays=SynchronizeAndCopyOwnedHost(arrays)
  writer=zeroed mz_zip_archive
  require mz_zip_writer_init_file(&writer,temp_path,reserve_size=0)
  for key in sorted schema key order:
    bytes=EncodeNPY(host_arrays[key])
    require mz_zip_writer_add_mem(&writer,key+'.npy',bytes,size,MZ_NO_COMPRESSION)
  require mz_zip_writer_finalize_archive(&writer)
  require mz_zip_writer_end(&writer)          # RAII also ends writer on failed paths
  reopen completed file and fsync; read back metadata/shapes/checksum before publish
```

路径是 run 模块在自身新运行目录中保留的唯一 temp 名称；失败保留 temp 标记，绝不覆盖已有 checkpoint。原子发布另见 run。原生/Python 往返测试必须覆盖所有上表 dtype、复数正负模、非零非连续 active 映射、Unicode、scalar shape、跨端字节序以及截断/重复条目；`allow_pickle=False`。这些测试尚需实现，库头文件可用不等于文件往返已通过。
