# CLAUDE 동작
니가 구사한 코드는 왠만한 건 다 파악 가능해. 주석은 나중에 한꺼번에 만들꺼야, 만들지마
함수나 메소드 가이드도 나중에 할꺼야.

CLAUDE.md에서 불필요하게 토큰을 낭비 하지 않도록, 작업 내역의 개요를 확인해라
작업 관리는 GitHub Issues로 한다. TODO.md 같은 파일은 만들지 마라.
Git 관련 내용(커밋 메시지, PR, 이슈 코멘트)은 영어로 작성한다.
커밋 메시지에 "Co-Authored-By" 넣지 마라. PR에 "Generated with Claude Code" 같은 광고성 메시지 넣지 마라.

코드 검증은 `tests/`에서 적절한 `.py`를 찾아 테스트 케이스를 추가하여 진행한다. `python -c` 같은 임시 실행은 하지 않는다.

## CLI 버전
- git 2.43.0
- gh 2.45.0

## gh CLI 주의사항
- `gh issue view <num>` 는 Projects Classic 지원 deprecated 경고로 exit code 1 반환 → **반드시 `--json` 플래그 사용**
  - 예: `gh issue view 40 --json title,body,comments`
- `--repo` 플래그 없이도 현재 디렉토리의 remote origin에서 자동 추론됨

# modeler 모듈 요약

## 아키텍처 개요
```
Project(path, cache_maxsize)          경로·캐시 소유, 프로젝트 전역 레지스트리
  ├─ PipelineBuilder ──build()──► Pipeline    가변 정의 → 불변 stage 그래프
  ├─ Collectors                     Collector 인스턴스 레지스트리
  ├─ TrialStore                     Trial 정의 + 실행 이력
  ├─ Experimenter(name)             CV 실험 (exp/{name}/)
  └─ Trainer(name)                  전체 데이터 학습 (trainers/{name}/)
                                       └─ to_inferencer() ──► Inferencer
```
- **Project** (`_project.py`): 디렉토리 레이아웃 소유 + `TrialStore`/`ExperimenterStore` 레지스트리. Pipeline 버전은 Project가 색인하지 않음 — 각 pipeline이 자기 db에 자기 버전을 직접 관리(`build_pipeline` 참조). **모든 컴포넌트가 단독 동작 가능(2026-08-01)** — Experimenter/Trainer도 이제 `project` 참조를 안 들고, `path`/`cache`/`experimenter_store`/이미 로드된 `pipeline` 객체만 생성자로 받음. `Project.experimenter()`/`trainer()`(및 `load_*`)가 `(pipeline_name, pipeline_version)` → `Pipeline` 객체 변환과 `cache`/`experimenter_store` 주입을 대신 해주는 다리 역할
- **PipelineBuilder / Pipeline** (`_pipeline.py`): 가변 빌더 + `build()`가 만드는 불변 **stage 전용** 그래프
- **Trial / make_trials** (`_trial.py`): 평가할 구성 하나 = 예전의 Head 노드. Pipeline 밖에 있음
- **TrialStore** (`_trial_store.py`): `trials`(정의) + `experiment_hist`(fold별 실행 이력)
- **Experimenter** (`_experimenter.py`): CV 실험 실행/관리
- **Trainer** (`_trainer.py`): 학습 실행/관리 (split 기반)
- **Inferencer** (`_inferencer.py`): 학습된 processor를 새 데이터에 적용
- **NodeStore** (`_store.py`): run 하나당 하나 — 노드 아티팩트(obj.pkl/result.pkl) + 실행 이력(`node_hist`, 구 NodeInfoStore 통합) 둘 다 소유
- **DataFlow / TrainDataFlow** (`_flow.py`): fold별 데이터 흐름 및 stage 빌드 (DataFlow는 NodeStore를 컴포지션으로 보유, outer_idx/inner_idx도 보유)
- **_executor.py**: `_execute_single`(단일 프로세스) + `_execute_multi`(멀티 워커) — 둘 다 stage/trial 공용, `collectors` 인자로 구분 — 실제 실행

## Node/Trial 상태 모델

### 3-State
`init → built` / `init → error → (reset) → init`

| 상태 | Disk | 설명 |
|------|------|------|
| **init** | - | 정의만 된 상태 |
| **built** | O | 빌드 완료, 결과 추출 가능 |
| **error** | info only | 실행 중 에러 발생, 내역 보존 |

- **finalized 상태 없음(2026-08-01 제거)**: obj/result를 지우고 메모리만 비우는 중간 상태였는데,
  Experimenter의 open/close 개념 자체가 없어지면서(아래) 존재 이유가 사라짐 — `NodeStore.finalize()`,
  `Experimenter.finalize()`/`reinitialize()`/`close_exp()`/`reopen_exp()` 전부 제거. 이제 아티팩트를
  없애려면 `reset_nodes()`(완전 삭제, `init`으로 되돌림)만 있음

### Experimenter 상태 없음(2026-08-01, open/close 제거)
Experimenter엔 이제 상태 게이트가 없음 — `build()`/`exp()`는 언제나 호출 가능. `open()`/`close()`/
`status` 속성/`experimenters` 테이블의 `status` 컬럼 전부 제거됨. Trial 아티팩트는 다음 `exp()`
호출까지 그대로 남아있는 게 기본값이고(`set_pipeline`이 Trial을 stale 취급 안 하는 것과
같은 이유 — 아래 "Staleness" 섹션 참조), 명시적으로 지우고 싶으면 `reset_nodes()`를 직접 호출

## 핵심 클래스

### Node 역할
- **DataSource** (`_DataSourceNode`, key=`None`): 원본 데이터 스키마 및 target 정의
- **Stage**: 전처리/변환 (TransformProcessor). **Pipeline에 담기는 유일한 종류**
- **Trial**: 모델링/예측 (PredictProcessor). Pipeline 밖 — `_trial.py` 참조

### PipelineBuilder / Pipeline 분리 (`_pipeline.py`)
```
PipelineBuilder  — 가변. grps 계층, SQLite(pipeline.db), set_grp/set_node
  └─ .build() ──► Pipeline  — 불변 스냅샷. grp 상속 해소 완료, 순수 데이터
```
- **Experimenter/Trainer/Inferencer는 `Pipeline`만 보유** — builder를 넘기면 `TypeError`(`_run_common.require_built_pipeline`). builder를 나중에 수정해도 진행 중인 실행에 새어 들어가지 않음
- **Pipeline은 stage 전용** — `role` 개념이 코드 어디에도 없음(2026-08-01 완전 제거). 예전엔 `_PipelineNode`/`_DataSourceNode`/`_BuiltNode`/`_BuiltDataSource`의 `get_attrs()`가 전부 `role`(`'stage'`/`'datasource'`)을 실었고 `Connector(role=)`가 그걸로 Stage와 Trial을 갈랐지만, Trial 쪽 `role`이 없어지면서(아래 "Trial" 섹션) 남은 소비자가 하나도 없어져 4곳 모두에서 삭제. `_BuiltDataSource.__slots__`의 `role`, `_BuiltNode.role` 클래스 상수도 같이 제거
- grp는 build를 넘어가지 않음 — 원래 그룹명은 표시용 `label`로만 남음 (`_BuiltNode.label`에만 남고, `ProcessorSpec`엔 `grp`/`label` 둘 다 안 실림)
- 노드에 `tag` 없음 (Trial로 이동)

#### PipelineBuilder
- `VAR_TYPES = frozenset({'numerical', 'ordinal', 'nominal', 'text', 'binary', 'datetime'})`
- **`_params_equal(a, b)`**: `a == b` 한 줄 — params가 순수 데이터/ref spec만 담도록 강제되므로 `__dict__` 재귀 비교 같은 우회가 불필요해짐
- `nodes`: `{name: _PipelineNode}` (`None` → `_DataSourceNode`), `grps`: `{name: _PipelineGroup}` (`'__datasource__'` 항상 존재)
- `datasource`: `nodes[None]` 반환 property
- `set_datasource(schema, targets=None)`: DataSource 스키마/target 설정
- `set_grp(exist='diff'|'skip'|'error'|'replace')`, `set_node(exist=...)`, `rename_grp`, `remove_grp`, `remove_node`
  - **`role` 파라미터 없음** (stage 전용), **`tag` 파라미터 없음**
  - **`processor`/`adapter`/`params` 스펙 검증** (`_validate_processor`/`_validate_adapter`/`_validate_params`) — 산 객체를 넘기면 `TypeError`. 아래 "Lazy resolution" 참조
- `build()` → `Pipeline`
- `get_node_names(query)`, `get_node_spec(name)`, `_find_descendants(name)`
- `sync()`: DB가 source of truth. 그룹/노드 필드를 직접 값 비교(`diff()`)해 갱신하고, **그룹이 바뀌면 그 그룹(+자식 그룹) 소속 노드들의 attrs 캐시도 함께 무효화**해 `changes['nodes']['updated']`에 포함시킴(노드 자신의 행은 안 바뀌었어도 상속받는 값이 바뀌었으므로)
- **`serial` 없음(2026-08-01 제거)**: 예전엔 정의 변경마다 새 UUID를 부여해 staleness/버전 판정에 썼지만, 지금은 두 용도 모두 다른 방식으로 대체됨 — staleness는 `Pipeline.diff_from`(아래)의 값 비교, 버전은 해시/dedup 없이 `PipelineStore`가 관리하는 단순 `max+1` 카운터(`Pipeline.content_key`도 없음 — 아래 "저장 구조" 참조)
- `copy()`, `copy_nodes(node_names)` — 선택적 복사 (builder→builder)
- `compare_nodes(nodes)` → `{processor_name: DataFrame}` (params 차이 + edges['X'] stage별 변수 차이)
- `desc_pipeline(max_depth, direction)`, `desc_node(node_name, direction, show_params)`: Mermaid 다이어그램 — grp 계층이 필요하므로 **builder 전용**

#### Pipeline (빌드 결과)
- `nodes`: `{name: _BuiltNode}` — `None` 키는 `_BuiltDataSource` (builder와 동일한 관례)
- `_BuiltNode` 속성(`__slots__`): `name`, `label`, `processor`, `edges`, `method`, `adapter`, `params`, `desc`, `output_edges` (`role` 없음 — 위 참조)
- `pipeline_id`(builder 신원) / `build_id`(빌드 호출마다 새 UUID) / `version`(`int | None`) — **`Project.build_pipeline()`이 저장할 때만 세팅**. `builder.build()`를 직접 부르면 `None`(미저장 in-memory 빌드)
- `get_node(name)`, `get_node_spec(name)`(ProcessorSpec — stage 전용, DataSource는 `pipeline.datasource`로), `get_node_names(query=None)`
- `topo_order()`: DataSource에서 내려오는 깊이순 노드명 (DataSource 제외) — 빌드 시 1회 계산해 캐시
- `descendants(name)`, `check_data_compatibility(data)`
- **`diff_from(old)`** → `set[str]`: 아래 "staleness" 섹션 참조
- `subset(node_names)`: 지정 노드 + 조상만 담은 새 Pipeline
- **불변성의 한계**: `params`/`edges`는 shallow copy — 중첩 값은 builder와 공유. "수정하지 않는다"는 관례로 지킴

- **`_DataSourceNode`** (`_PipelineNode` 서브클래스):
  - `schema`: `{col: var_type}` — var_type은 VAR_TYPES 중 하나
  - `targets`: `list[str]` — 타겟 컬럼 목록 (타입과 별도)
  - `get_attrs(grps=None)`: `name`, `grp`, `schema`, `targets` **dict** 반환 — DataSource는 실행 대상이 아니라 `ProcessorSpec`을 만들 게 없어서 `get_spec`을 **오버라이드하지 않음**(같은 이름의 다형 메소드가 서로 다른 모양을 반환하는 걸 피하려고 아예 이름을 분리)

- **`_PipelineGroup`**: 노드 그룹 — builder 내부 전용
  - 속성: `name`, `processor`, `edges`, `method`, `parent`, `adapter`, `params`, `desc`
  - `children`: 자식 그룹명 리스트, `nodes`: 소속 노드명 리스트
  - `get_attrs(grps)`: 상위 그룹 속성 병합하여 **dict** 반환 (`desc`는 상속 안 됨, 각 요소 독립) — 그룹은 실행 단위가 아니라 상속 해소용이라 `ProcessorSpec`이 아님(캐시 `self.attrs`/`update_attrs()`도 그대로)
  - `diff(processor, edges, method, parent, adapter, params)`: 달라진 필드명 리스트 반환 (`desc` 제외 → desc-only 변경은 rebuild 미유발)

- **`_PipelineNode`**: 개별 노드 — builder 내부 전용
  - 속성: `name`, `grp`, `processor`, `edges`, `method`, `adapter`, `params`, `desc`
  - `output_edges`: 이 노드를 입력으로 사용하는 노드명 리스트
  - `get_spec(grps)`: 그룹 속성과 노드 속성을 병합해 `ProcessorSpec` 반환(캐시는 `self.spec`, 무효화는 `update_spec()`)
  - `diff(grp, processor, edges, method, adapter, params)`: 달라진 필드명 리스트 반환 (`desc` 제외)
  - `set_grp`/`set_node`: `desc` 파라미터 수락; exist='diff' skip 경로에서도 `desc`는 업데이트됨

- **ColSelector** (`_pipeline.py`): processor params(예: `cat_features`, `cat_cols`)에 쓰는 지연(lazy) 컬럼 선택자
  - `__init__(dsl_string='*')` — DSL 문자열 하나만 보유(정의 시점엔 데이터 불필요, `edges[key]`와 동일한 원칙)
  - **params에는 인스턴스가 아니라 ref-dict로 지정**: `{"__ref__": "mllabs.ColSelector", "__params__": {"dsl_string": "*@categorical"}}` (인스턴스는 `set_grp`/`set_node`가 `TypeError`로 거부)
  - `_node_processor`가 Processor 생성 시 `resolve_ref_values()`로 인스턴스화하고, fit 시점에 `_resolve_col_selectors`가 `eval_expr(parse(v.dsl_string), data)`로 컬럼 확정

### ProcessorSpec (`_pipeline.py`, 2026-08-02)
**Stage와 Trial이 공통으로 resolve되는 단 하나의 실행 단위 표현.** 예전엔 `get_attrs()`가 dict를 돌려줬는데, 클래스로 바꾸면서 필드가 정확히 6개로 고정됨:
`name`, `processor`, `edges`, `method`, `adapter`, `params` (`__slots__`, immutable 취급, 값 기반 `__eq__`)

- 이 중 **5개(`name`/`processor`/`method`/`adapter`/`params`)가 Processor 생성자 인자 그대로** — `_node_processor.py`의 `TransformProcessor`/`PredictProcessor`가 받는 것과 1:1
- **`edges`는 Processor한테 안 넘어감** — flow가 "무엇을 먹일지" 정하는 입력 배선이고, 실행 시점에 실제 데이터에 대해서만 lazily 컬럼으로 확정됨. 그래서 이름이 `ProcessorAttr`가 아니라 `ProcessorSpec`("아직 resolve 안 된 선언"이라는 뜻으로 이 모듈이 이미 쓰고 있는 어휘)
- **표시 전용 필드는 일부러 뺌** — Stage의 `label`(원래 grp명), Trial의 `desc`/`tag`. 전부 원본 객체에서 직접 꺼낼 수 있고, attrs dict 시절에도 이 경로로는 아무도 안 읽었음(제거 전 전수 조사로 확인)
- 만드는 쪽: `_BuiltNode.get_spec()`, `_PipelineNode.get_spec(grps)`(그룹 상속 해소 + `self.spec` 캐시/`update_spec()`), `Trial.get_spec()`
- 쓰는 쪽: `Job.spec` → `_process()`(Processor 생성) / `_definition_of()`(staleness·info) / flow의 입력 준비 / `Connector.match(spec)` / `_describer` / `Inferencer.node_specs`
- **DataSource는 여기 안 들어감** — `_DataSourceNode.get_attrs(grps=None)`/`_BuiltDataSource.get_attrs()`는 `name`/`grp`/`schema`/`targets` **dict**를 그대로 반환. 실행 대상이 아니라 만들 Processor가 없어서, `get_spec`을 오버라이드하지 않고 이름을 아예 분리함(같은 이름의 다형 메소드가 서로 다른 모양을 반환하는 걸 피함). 접근은 `pipeline.datasource` / `builder.datasource`
- `_PipelineGroup.get_attrs(grps)`도 dict 그대로 — 그룹은 실행 단위가 아니라 상속 해소 중간 단계라 `ProcessorSpec`이 아님

### Trial (`_trial.py`)
Head를 Pipeline에서 떼어낸 결과. **Experiment 클래스는 없음** — Trial 리스트를 직접 넘긴다.

- **`Trial`**: 평가할 구성 하나. `name`, `processor`, `method`, `adapter`, `params`, `edges`, `desc`, `tag`
  - `desc`(2026-08-01, `label`에서 이름 변경): 순수 표시용 설명 문자열 — `PipelineBuilder`의 `desc`와 같은 역할(매칭/diff/저장 식별에 전혀 관여 안 함). 예전 `label`은 `make_trials()`가 `label=name`으로 자동 채워 sweep 그룹핑 용도로도 썼는데, "설명"이라는 새 의미와 안 맞아서 그 자동 채움은 제거 — `desc`는 명시적으로 안 주면 그냥 `None`
  - `get_spec()`: Stage의 `Pipeline.get_node_spec()`과 **똑같은 `ProcessorSpec`**을 반환. `role` 키는 2026-08-01에 제거 — Connector가 Trial과 Stage를 구분할 목적으로 쓸모가 없었음(Collector는 애초에 Trial job에만 붙고 Stage job엔 안 붙으므로 `role='head'`로 걸러야 할 대상이 없었음). `Connector(role='head')`로 필터하던 곳은 이제 `Connector()`(무필터)로
  - **이름이 식별자**. 디스크 아티팩트 디렉토리명이자 `TrialStore`(`trials` 테이블 PK)의 키 — 재정의하면 아티팩트도 `TrialStore` row도 덮어씀
  - **`content_key()` 없음(2026-08-01 제거)**: 정의를 값으로 비교하는 유틸이었으나 실제 호출부가 전혀 없었음(저장/식별 어디에도 안 쓰임 — `TrialStore.has()`는 애초부터 필드별 직접 비교)
  - `stage_names()`: edges가 참조하는 stage 이름 집합

- **`make_trials(name, processor, edges, method, adapter, params, param_grid, tags)`** → `list[Trial]`
  - `params`(전 trial 공통) + `param_grid`(`{param: [values]}`) 카테시안 곱, grid 키 정렬 기준 결정적 순서
  - 이름: 단일이면 `{name}`, 복수면 `{name}_{idx}` (0 패딩)
  - `_validate_processor`/`_validate_adapter`/`_validate_params`로 spec 검증 (Pipeline과 동일 규칙)

### Project (`_project.py`)
디렉토리 레이아웃 소유 + 프로젝트 전역 레지스트리. **모든 컴포넌트가 단독 동작 가능(2026-08-01)** —
Experimenter/Trainer도 `project` 참조를 안 들고 `path`/`cache`/(Experimenter는)`experimenter_store`/
이미 로드된 `pipeline` 객체만 생성자로 받으므로, Project는 이제 순수 "이 조각들을 짜맞춰주는 팩토리"일
뿐 두 클래스의 필수 의존성이 아니다. `experimenter()`/`trainer()`(및 `load_*`)가 하는 일:
1. `pipeline_name`/`pipeline_version`을 `load_pipeline()`으로 실제 `Pipeline` 객체로 변환
2. `cache`(및 Experimenter라면 `experimenter_store`)를 주입
3. 나머지 kwargs와 함께 `Experimenter(path, name, data, ...)`/`Trainer(path, name, data, ...)`를 호출

- `Project(path, cache_maxsize=4GB)` — `DataCache`를 소유하고 모든 Experimenter/Trainer가 공유
- 경로: `pipeline_path(name)`, `exp_path(name)`, `trainer_path(name)`, `inferencer_path(name)`, `collectors_path()`
- 팩토리: `pipeline_builder(name)`, `collectors()`, `experimenter(name, data, pipeline_name=, pipeline_version=, **kw)`, `load_experimenter(name, data, data_key=, aug_data=)`, `trainer(name, data, pipeline_name=, pipeline_version=, **kw)`, `load_trainer(name, data, aug_data=)`
  - `load_experimenter`/`load_trainer`가 예전엔 `Experimenter.load(project, ...)`/`Trainer.load(project, ...)`에 위임했지만, 그 재구성 로직 자체가 이제 여기로 옮겨옴(meta/`__splitters.pkl`/`__trainer.pkl` 읽기, `load_pipeline` 호출) — Experimenter는 project-aware `load()`가 아예 없어졌고, Trainer의 `load()`는 `(path, data, save_data=, cache=, pipeline=, aug_data=)`만 받는 project-agnostic 클래스메소드로 남음(`Trainer._read_save_data(path)`로 파일을 먼저 읽어 `pipeline_version`을 얻은 뒤 `load_pipeline`으로 객체를 만들어 넘겨줌 — 파일을 두 번 안 읽으려고 `save_data`를 인자로 전달)
- **Pipeline 버전**: `build_pipeline(builder)` → `builder.build()` 호출 후 결과를 다음 버전(1부터, `builder._store`가 관리하는 카운터의 `max+1`)으로 저장하고 `pipeline.version`에 세팅해 반환. **content dedup 없음** — 내용이 같아도 호출할 때마다 새 버전(`builder`에 path가 없으면 `ValueError`)
  - 카운터/버전 파일은 **Project가 아니라 각 pipeline 자신의 db**(`pipelines/{name}/{name}.db`)가 소유 — `build_pipeline`은 `builder._store.save_version()`에 위임할 뿐, 프로젝트 전역 색인이 없음
  - `load_pipeline(name, version=None)`, `list_pipeline_versions(name)` — 둘 다 내부적으로 `PipelineStore(pipeline_path(name), name)`를 통해 조회
  - 저장은 pkl (`v{n}.pkl`) — 형식은 `PipelineStore.save_version`/`load_version` 뒤에 숨어 있어 나중에 교체 가능
- `trials`: `TrialStore`, `experimenters`: `ExperimenterStore`, `list_experimenters()`

### ArtifactStore (`_store.py`, 공통 인터페이스, 2026-08-01)
`NodeStore`/`TrialStore`가 공유하는 메소드 모양의 base class. 두 그룹으로 나뉨:
- **아티팩트** (`write_objs`/`write_obj`/`write_result`/`get_objs`/`get_obj`/`get_result`/
  `list_nodes`/`status`/`reset_node`) — **`NodeStore`만 전부 구현**. `TrialStore`는
  `ArtifactStore`를 상속만 할 뿐 하나도 오버라이드하지 않음 — Trial의 obj/result는 (Stage와 같은
  디렉토리를 쓰는) 그 run의 `NodeStore`가 갖고, `TrialStore`는 정의(`trials`)+실행 이력
  (`experiment_hist`)만 가지므로 쌓을 obj/result 자체가 없음. 상속만 해두는 이유는 base의 기본
  동작이 `NotImplementedError`를 던지는 것이라 — `TrialStore`에서 이 메소드들을 호출하면 (없는
  메소드라 `AttributeError`가 나는 대신) 의도가 분명한 `NotImplementedError`가 남
- **히스토리** (`record`/`get_hist`/`get_status`/`get_info`/`remove_hist`) — 둘 다 실제로
  구현(각자 자기 테이블에 대해). 모양만 여기 문서화해두는 것 — `TrialStore`가 (이미 한 run에
  스코프된 `NodeStore`엔 없는) experimenter 이름을 하나 더 키로 쓰기 때문에 override 시그니처가
  서로 다름(`record`/`get_hist` 등은 base에서 `*args, **kwargs`로만 선언)

### TrialStore (`_trial_store.py`)
```sql
trials(name PK, desc, processor, method, adapter, params, edges, tag)
experiment_hist(trial_name, experimenter, outer_idx, inner_idx,  -- PK
                pipeline_version, status)
```
- **인조식별자도 content hash도 없음.** 두 테이블 다 **이름이 PK**(`trials`는 trial 이름 하나, 이력은 trial 이름 + experimenter 이름). `pipeline_version`은 해시가 아니라 **정수** — 그 실행의 `Experimenter.pipeline_version`을 그대로 기록
- 이름으로 키잉하는 이유: 아티팩트가 이미 이름으로 키잉돼 있음(`{exp}/__folds/{o}/{i}/{trial_name}/`, `{project}/exp/{name}`). 맞춰두면 조인 없이 읽히고, **정의를 바꿔 재실행 = 아티팩트 덮어쓰기 = 행 덮어쓰기**가 두 테이블 모두 일관됨(`register`는 `INSERT OR REPLACE`)
- **content_key 컬럼 없음(2026-08-01 제거)**: params가 평문 데이터로 강제된 덕에 정의 일치 여부는 값 비교 하나로 충분(`has()`) — 해시 컬럼은 이걸 재서술할 뿐이었음. `experiment_hist`는 실행 로그일 뿐 정의의 출처가 아니라서, 이름이 재정의되면 예전 정의 자체를 복원하는 기능은 애초에 없음(`Trial.content_key()` 메소드 자체도 실호출부가 없어 2026-08-01에 완전히 삭제됨 — 값 비교 유틸이 존재하는 것과 그걸 실제로 쓰는 코드가 있는 것은 별개였음)
- **아티팩트 rebuild 필요 여부는 `info['definition']`이 아니라 `experiment_hist`가 판정(2026-08-01 개정)**: `Experimenter._make_jobs`는 더 이상 디스크의 `info['definition']`을 트리거의 정의와 비교하지 않음 — `experiment_hist`에서 그 fold의 `status`만 확인해서 `'built'`면 스킵, `'error'`거나 기록이 없으면 job을 만듦. 즉 trial을 재정의해도 이미 `'built'`로 기록된 fold는 자동으로 재실행되지 않음(원하면 `reset_nodes`로 명시적으로 지워야 함) — `NodeStore`를 아예 들여다보지 않음
- `register(trial)`/`register_all(trials)`: 이름 기준 upsert(반환값 없음). `has(trial)`: 그 이름에 저장된 게 **지금** 이 정의와 같은지 필드별 비교. `get_by_name(name)`, `list_trials()`
- `record(trial_name, experimenter, outer_idx, inner_idx, pipeline_version, status)`, `get_hist(...)`, `get_status(...)`, `remove_hist(...)`

### Experimenter (`_experimenter.py`)
- **Project 의존성 없음(2026-08-01)** — 생성자: `Experimenter(path, name, data, ..., cache=None, experimenter_store=None, pipeline=None, pipeline_name='pipeline', _save=True)`. `path`는 이미 존재해야 하는 이 run의 base 디렉토리(없으면 생성자가 직접 mkdir) — 보통은 `project.experimenter(name, data, pipeline_name=, pipeline_version=)`로 생성하고, Project가 `(pipeline_name, pipeline_version)` → `Pipeline` 객체 변환 + `cache`/`experimenter_store` 주입을 대신 해줌
- **이름이 식별자**: 경로는 `{project}/exp/{name}`, `TrialStore` 이력의 키도 이 이름. `exp_id` 같은 UUID 없음
- **Pipeline은 객체로 지정** — `set_pipeline(pipeline, pipeline_name=None)`이 이미 로드된 `Pipeline`을 그대로 받아 채택. 이 클래스는 이름/버전으로 파이프라인을 **로드할 방법 자체가 없음**(project 참조가 없어서) — 버전 번호로 지정하고 싶으면 `Project.experimenter(..., pipeline_version=)`을 쓸 것. `self.pipeline_version`은 별도로 안 들고 `pipeline.version`에서 그대로 읽음(단일 출처). `pipeline.pkl`을 실험 디렉토리에 복사하지 않고 **포인터(`pipeline_name`, `pipeline_version`)만** 저장
  - 버전 전환 시 `pipeline.diff_from(self.pipeline)`으로 stale 판정 → `reset_nodes()`로 해당 stage 아티팩트만 제거. Trial은 건드리지 않음(아래 "Staleness" 섹션 참조)
- `cache`(`DataCache`, optional)/`experimenter_store`(`ExperimenterStore`, optional) — 둘 다 `None`이면 그 기능만 조용히 꺼짐(`cache=None`이면 캐시 없이 동작, `experimenter_store=None`이면 `_save()`가 meta 저장을 스킵) — standalone 사용을 위한 graceful degradation
- **상태 게이트 없음(2026-08-01, open/close 제거)**: `build()`/`exp()`는 언제나 호출 가능. `finalize`/`reinitialize`/`close_exp`/`reopen_exp`/`open`/`close`/`set_status`/`status` 속성 전부 제거됨
- **OS log capture** (`open_os_log`/`close_os_log`/`os_log`) — 위 상태 게이트와 무관한 별개 기능:
  - `open_os_log(log_path=None)`: 이 프로세스의 OS-level stdout/stderr(fd 1/2)를 `{path}/__worker_logs/master.log`(기본값)로 dup2 리다이렉트 시작 — `self._os_log_state`에 원본 fd/`sys.stdout`·`stderr` 백업 보관. 이미 open이면 에러
  - `close_os_log()`: 리다이렉트 원복(`sys.stdout`/`stderr` 및 fd 1/2 복구). open 안 된 상태에서 호출하면 no-op
  - `os_log(log_path=None)`: 위 둘을 감싼 컨텍스트 매니저 — `with e.os_log(): e.build(n_jobs=1); e.exp(n_jobs=4)`
  - open~close 구간 동안: `n_jobs=1`인 `build`/`exp`는 같은 프로세스에서 돌기 때문에 마스터 리다이렉트가 그대로 캡처(별도 처리 불필요). `n_jobs>1`이면 그 구간에 한해 `log_dir`이 전달되어 워커별 리다이렉트도 같이 동작(위 `build`/`exp` 항목 참조)
  - `sys.stdout`/`stderr`는 원본 fd의 dup으로 rebind되므로, capture가 열려 있어도 `DefaultLogger`의 진행률 표시 등 Python 레벨 출력은 그대로 콘솔에 보임 — dup2로 fd 1/2만 로그 파일로 돌리기 때문에 native(C-level) 직접 write만 잡힘
- **pipeline 필요** (`_require_pipeline()`로 미설정 시 에러):
  - `build(nodes=None, rebuild=False, n_jobs=1, gpu_id_list=None, logger=None)` — stage 빌드
  - **`exp(trials, trial_store, collectors=None, n_jobs=1, gpu_id_list=None, logger=None)`(2026-08-01, `trial_store` 필수 인자로 추가)**
    - `trials`: **`[(Trial, outer_idx, inner_idx), ...]`** 튜플 리스트. fold 전개를 여기서 하므로 executor는 목록을 그대로 실행
    - `trial_store`(`TrialStore`): 필수 — Trial 정의 등록/fold 스킵 판정/이력 기록 전부 여기로 함(호출부가 넘겨야 함, 보통 `project.trials`)
    - `collectors`: `Collectors` 레지스트리 / Collector 인스턴스 리스트 / `None`
    - `_make_jobs(trials, trial_store)`가 `Job(name, spec, outer_idx, inner_idx, flow, need_gpu)` 리스트를 만듦(Stage/Trial 공용 클래스 — 아래 `_executor.py` 섹션 참조). skip 판정은 `trial_store.get_status(name, self.name)`(= `TrialStore.experiment_hist`)의 fold별 status로만 함 — `'built'`면 스킵, 그 외(`'error'`/기록 없음)면 job 생성. GPU 판정도 여기서 하고, adapter resolve는 **trial 이름당 1회**
    - Trial 정의를 *trial_store*에 등록하고, `TrialHistTracker`가 fold별 done/error를 이력에 기록
  - `n_jobs`는 실제 작업 수로 상한 처리 (`min(n_jobs, len(jobs))`) — 유휴 워커/progress bar 방지
  - `n_jobs > 1`이고 OS log capture가 open일 때만 워커 stdout/stderr를 `{path}/__worker_logs/worker_{i}.log`로 리다이렉트
  - `get_node_info()`: 노드 요약 Markdown
- **pipeline 불필요** (디스크 상태만으로 동작): `get_status(node_name)`, `reset_nodes(nodes)`, `show_error_nodes(nodes=None, traceback=False, trial_store=None)`(trial_store 없으면 Stage 에러만 보고), `get_objs(node_name, outer_idx=0, inner_idx=0)`
  - fold당 `NodeStore`가 **하나**(`train_data_flows[j]`)라 예전의 이중 store stale 캐시 문제는 없음
- **OS log capture** (`open_os_log`/`close_os_log`/`os_log`) — 위 상태 게이트와 무관한 별개 기능:
  - `open_os_log(log_path=None)`: 이 프로세스의 fd 1/2를 `{path}/__worker_logs/master.log`로 dup2 리다이렉트
  - `close_os_log()`: 원복. `os_log()`는 둘을 감싼 컨텍스트 매니저
  - `sys.stdout`/`stderr`는 원본 fd의 dup으로 rebind되므로 진행률 등 Python 레벨 출력은 콘솔에 그대로 보임 — native(C-level) write만 잡힘
- `get_worker_logs(worker=None)`: 캡처된 네이티브 출력 — `{worker_idx: text, 'master': text}`. 매 실행마다 덮어씀
- `get_train_data(edges, o_idx=0, i_idx=0)` / `get_valid_data(...)` / `get_test_data(...)`: 출력 추출 헬퍼
- `aug_data`: 외부 데이터를 DataSource 수준에서 inner train split에 append — 미퍼시스트
- 저장/로드: `project.load_experimenter(name, data, data_key=None, aug_data=None)` — **Experimenter 자신은 project-aware `load()`가 없음**(2026-08-01 제거). meta 조회/`__splitters.pkl` 읽기/`load_pipeline` 호출까지 전부 `Project.load_experimenter()` 안에서 하고, 그 결과로 얻은 `pipeline` 객체를 보통 생성자에 그대로 넘김
  - meta는 **프로젝트 전역 `experimenters.db`**에 (`_experimenter_store.py`) — `name`이 PK, 타입 있는 컬럼(`data_key, title, pipeline_name, pipeline_version`). 실험 디렉토리에 `__exp.db`는 없음
  - splitter 객체(`sp, sp_v, splitter_params`)는 ref-직렬화 불가라 `{exp_path}/__splitters.pkl`에 pickle

### DataCache (`_cache.py`)
- `cachetools.LRUCache` 기반, 용량(bytes) 단위 관리. `Project`가 소유(`project.cache`)해서 모든 Experimenter/Trainer가 공유
- 키가 **`(scope, node, typ)`**(2026-08-01 개정, `outer_idx`/`inner_idx` 제거) — `scope`는 그 fold를 만든 `TrainDataFlow`가 **자기 생성자에서 만드는 랜덤 id**(`self.scope = uuid.uuid4().hex`), `str(store.path)`가 아님. 이전엔 경로 문자열을 `scope`로 썼는데, `Path(path)`를 `resolve()` 안 해서 — Experimenter/Trainer를 Project 없이 독립 생성하고 `cache=`를 외부 주입받을 수 있게 된 뒤로는 — 서로 다른 물리 디렉토리가 우연히 같은 상대경로 문자열이 되어 충돌할 여지가 실제로 있었음(발견된 문제, 이번에 해소)
  - **`TrainDataFlow` 인스턴스 하나 = 정확히 그 (run, fold) 하나**(fold마다 새로 만들고 공유 안 함)라, 그 인스턴스 자신의 랜덤 id만으로 이미 "이 run의 이 fold"가 유일하게 식별됨 — 그래서 `outer_idx`/`inner_idx`를 키에 또 넣을 필요가 없어짐(중복 제거). 트레이드오프: 리로드(`Project.load_experimenter` 등, 새 Python 인스턴스 = 새 scope id)하면 이전 인스턴스가 캐싱해둔 항목은 다시 못 만남 — cache miss일 뿐 잘못된 값이 나오는 게 아니라 허용 가능한 손실로 판단
- `get_data(scope, node, typ)`, `put_data(scope, node, typ, data)`
- `clear_nodes(nodes)`: 특정 노드들의 캐시 삭제(이름만 매칭 — scope 무관하게 지움. 여러 run이 같은 노드 이름을 우연히 쓰면 서로의 캐시까지 같이 지워짐 — 안전하지만(재계산될 뿐) 낭비이긴 함, 아직 미해결)

### NodeStore (`_store.py`)
- `ArtifactStore`(위 섹션 참조)를 상속해 그 아티팩트 메소드들을 전부 구현
- **run 하나(Experimenter/Trainer 하나)당 인스턴스 하나(2026-08-01, NodeInfoStore와 통합)** — 예전엔 fold 하나당 인스턴스 하나였는데, 이제 `outer_idx`/`inner_idx`를 매 호출마다 받아서 그 fold의 경로/이력을 그때그때 계산. Experimenter/Trainer가 각자 자기 base path(`exp/{name}`, `trainers/{name}`)로 생성자에서 한 번만 만들고, 그 run의 모든 fold가 같은 인스턴스를 공유
- 아티팩트: `{path}/{outer_idx}/{inner_idx}/{node_name}/`
  - `obj.pkl` — processor 객체, `result.pkl` — fit_transform/fit_predict 출력. **info.pkl 없음**(status/definition/fit_time/edges/train_shape/error 전부 안 남음)
  - `node_path(name, outer_idx, inner_idx)`가 경로 조립. `write_objs(name, outer_idx, inner_idx, obj, result)`/`write_obj(name, outer_idx, inner_idx, obj)`/`write_result(name, outer_idx, inner_idx, result)` — 전부 일반 instance 메소드로 `node_path()`를 내부에서 호출해 경로를 조립함(2026-08-01, staticmethod에서 전환) — get_*/status/reset_node와 시그니처가 통일됨. 서브프로세스 워커는 살아있는 인스턴스가 필요 없는 대신, 이제 스폰 시점에 이 `store` 인스턴스 자체를 넘겨받음(`NodeStore`는 열린 커넥션을 `self`에 들고 있지 않아 picklable — `ProcessWorker(conn, collectors, store, ...)` 참조)
  - `get_objs(name, outer_idx, inner_idx)`/`get_obj(...)`/`get_result(...)`/`list_nodes(outer_idx, inner_idx)`/`status(name, outer_idx, inner_idx)`(`None`/`'built'` — obj.pkl 존재만 봄, `'error'`는 여기서 절대 안 보임)/`reset_node(name, outer_idx, inner_idx)` — 전부 instance 메소드, 인자로 fold를 받음
- **history(구 NodeInfoStore, 여기 통합됨)**: SQLite `node_hist(node_name, outer_idx, inner_idx, pipeline_version, status, info)` — PK는 `(node_name, outer_idx, inner_idx)`뿐(run_name 컬럼 없음 — store 자체가 이미 그 run에 스코프돼 있어서 불필요)
  - `record(name, outer_idx, inner_idx, pipeline_version=, status=, info=)` — `info`는 `status` 제외 나머지 전부(`build_id`, `definition`, `fit_time`, `edges`, `train_shape`, `warnings`, 실패 시 `error`) JSON 인코딩, `NodeInfoTracker`(`_tracker.py`)가 기록
  - `get_hist(node_name=, outer_idx=, inner_idx=, pipeline_version=)`(각 필터 optional)/`get_status(node_name)`/`get_info(node_name)`(`{(outer_idx, inner_idx): ...}`)/`get_fold_info(outer_idx, inner_idx)`(fold 하나의 `{node_name: info}` 전체 — `DataFlow.load()`가 씀)/`remove_hist(node_name=)`
  - `'error'`는 오직 여기서만 보임 — `Experimenter.show_error_nodes`/`Trainer.get_node_error`가 조회

### DataFlow / TrainDataFlow (`_flow.py`)
- **DataFlow(2026-08-01, NodeStore 상속 → 컴포지션 전환 + outer_idx/inner_idx를 여기로 이동)**: 생성자가 `NodeStore` 인스턴스(`self.store`, run 전체가 공유) + 이 fold의 `outer_idx`/`inner_idx`를 받음. `status`/`get_obj`/`get_objs`/`get_result`/`list_nodes`/`node_path`는 `self.store.X(name, self.outer_idx, self.inner_idx)`로 위임하는 얇은 메소드. `reset_node`만 위임 + `node_objs`/`_node_edges`에서도 같이 지우는 조합 동작
  - `node_objs`: `{name: (obj, result)}`(info 없음), `_node_edges`: `{name: edges}`
  - **`load()`가 `self.store.get_fold_info(self.outer_idx, self.inner_idx)`를 한 번 조회**해서 `edges`까지 복원한 뒤 `load_objs(name, edges=...)`. history에 행이 없는 노드는 로드 안 함(안전한 기본값) — Trial도 이 규칙으로 자연히 걸러짐(Trial 결과는 `TrialStore.experiment_hist`에만 기록되고 이 run의 `node_hist`엔 절대 안 들어가서, 학습된 모델이 메모리로 딸려 들어오지 않음). 예전의 `role == 'head'` 명시적 스킵은 2026-08-01 `role` 전면 제거와 함께 삭제
  - `get_data(source_data, edges)` → `{key: data}`
- **TrainDataFlow** (DataFlow 상속): stage 빌드 기능 추가. 생성자는 `store`(run이 공유하는 NodeStore)를 그대로 받아 `super().__init__(store, outer_idx=, inner_idx=)`로 넘김 — 더 이상 `path`를 안 받고, fold별로 자기 NodeStore를 새로 안 만듦
  - `data_source`: DataWrapperProvider (train/valid/**test** 제공 — `test_idx` 보유)
  - `outer_idx`/`inner_idx`는 NodeStore 키(아티팩트 경로, history row) — **DataCache 키에는 더 이상 안 들어감**(위 "DataCache" 섹션 참조, `self.scope`가 생성자에서 만드는 랜덤 id로 대체). **Trainer도 자연스러운 `(split_idx, 0)`을 그대로 씀**, 예전처럼 음수 offset이나 별도 `info_fold` 개념 없음
  - `get_train(edges)`, `get_valid(edges)`, **`get_test(edges)`** — flow 하나로 job의 모든 입력을 만들 수 있어야 `Job`이 자족적이 됨
  - `set_objs(name, obj, result, info)`(현재 fit의 즉석 info에서 `edges`만 추출 — 디스크/history 안 거침), `get_missing_stages(pipeline)`

### Trainer (`_trainer.py`)
- **Project 의존성 없음(2026-08-01)** — 생성자: `Trainer(path, name, data, splitter=None, splitter_params=None, aug_data=None, cache=None, pipeline=None, pipeline_name='pipeline')` — 보통 `project.trainer(name, data, pipeline_name=, pipeline_version=)`로 생성, Project가 버전→객체 변환 + `cache` 주입을 대신 해줌(Experimenter와 동일한 패턴 — 위 "Project" 섹션 참조)
- 경로 `{project}/trainers/{name}`
- **`set_pipeline(pipeline, pipeline_name=None)`(2026-08-01, `set_pipeline_version`에서 이름 변경)**: 이미 로드된 `Pipeline` 객체를 받음(버전 번호 아님) — `self.pipeline_version`은 `pipeline.version`에서 그대로 읽음. Experimenter의 `set_pipeline`과 동일 이유로 이름을 바꿈 — 버전 전환 시 `diff_from`으로 stale 제거
- **`set_trials(trials)`**: 학습할 Trial 리스트 + 그것들이 읽는 stage를 자동 선택(`_recompute_selection`). **Trial은 영속화되지 않음** — 로드 후 다시 호출해야 함
- `trials`, `trial_names()`, `trial_attrs()`, `selected_stages`
- `train_folds`: `[TrainFold]` — split당 `TrainDataFlow` 하나
- `train(n_jobs=1, gpu_id_list=None, logger=None)`: stage 먼저(위상 순서), 그 다음 Trial `Job` 실행
- `get_status(node_name)`: `train_data_flows[0]` 조회
- `process(data, v=None)`: generator, split마다 head output을 `v`(DSL 문자열)로 필터 후 concat하여 yield
- `to_inferencer(v=None)`: 학습된 Processor를 추출하여 Inferencer 생성
- `reset_nodes(nodes)`: 하위 종속 노드 포함 초기화
- 저장/로드: `save()`, `project.load_trainer(name, data, aug_data=None)` — `Trainer.load()` 자신은 project-agnostic 클래스메소드로 남아 `(path, data, save_data=None, cache=None, pipeline=None, aug_data=None)`만 받음(`save_data`는 이미 읽은 `__trainer.pkl` dict — 안 주면 `Trainer._read_save_data(path)`로 읽음). `Project.load_trainer()`가 `_read_save_data`로 먼저 읽어 `pipeline_version`을 얻고 `load_pipeline`으로 객체를 만든 뒤 `save_data`와 함께 `Trainer.load()`에 넘겨(파일 재독 방지) — `{path}/__trainer.pkl`엔 splitter/split_indices + `(pipeline_name, pipeline_version)` 포인터. Trial은 미저장

### Inferencer (`_inferencer.py`)
- 생성자: `(node_specs, selected_stages, selected_heads, n_splits, node_objs, v=None)`
- **Pipeline 의존성 없음** — `node_specs`(`{name: ProcessorSpec}`)만 보유. 실제로 필요한 건 `spec.edges`뿐이라 배포 아티팩트가 가벼움
- `node_objs`: `{name: [processor_split0, processor_split1, ...]}` — Processor 리스트 (Trainer 독립)
- `process(data, agg='mean', nodes=None)`: split 결과 자동 집계
  - `agg`: `'mean'`/`'mode'`/callable/`None`(list 반환). 단일 split이면 집계 없이 반환
  - `nodes`: str/list — 출력할 head 노드 선택 (None=전체). 미등록 노드 지정 시 ValueError
- 저장/로드: `save(path)`, `load(cls, path)` — 단일 `__inferencer.pkl`에 node_objs 포함

### Connector (`_connector.py`)
- `__init__(node_query=None, edges=None, processor=None)` — 3요소 선택적 매칭
- **`role` 파라미터 없음(2026-08-01 제거)**: Trial과 Stage를 구분할 목적으로 있었는데, Collector는 애초에 Trial job에만 붙고 Stage job엔 안 붙어서(`Experimenter.build()`가 Stage 쪽엔 Collector를 아예 안 넘김) 걸러야 할 대상 자체가 없었음 — 죽은 파라미터. `Trial.get_spec()`도 같은 이유로 `role`을 안 실음(위 "Trial" 섹션 참조)
- `processor`: **`"module.ClassName"` 문자열만** (클래스 인스턴스 아님) — resolve 안 하고 그대로 저장
- `match(spec)`: `ProcessorSpec`을 받아 설정된 요소만 검사, 모두 충족 시 True (2026-08-02, dict → ProcessorSpec)
  - node_query: str(regex) 또는 list(in), processor: `spec.processor`(문자열, Pipeline도 항상 문자열로 저장)와 **문자열 그대로 비교**(정규화 없음 — `set_grp`/`set_node`에 준 것과 같은 문자열 형태를 넘겨야 매칭)
  - edges: `{key: dsl_string}` — 각 key에 대해 노드의 resolved `edges[key]` 문자열과 **정확히 일치**해야 함 (contain 기반 아님)

### Collector (`collector/` 패키지)
- **Collectors** (`_registry.py`): Collector 인스턴스를 소유하는 레지스트리. `Project.collectors()`로 얻음
  - `Collectors(path=None)` — path 있으면 등록 시 `{path}/{name}`이 기본 저장 위치
  - `set_collector(name, collector, connector, path=None, params=None, exist='skip')` — 부품에서 조립. `collector`는 클래스 또는 `"module.ClassName"`, `connector`는 인스턴스 또는 `{__ref__}`, `params`엔 `resolve_ref_values` 적용
  - `get_collector`/`remove_collector`/`names()`/`in`/`len`/`iter`
  - **`resolve(names)`**: 미등록 이름이면 `KeyError` — 조용히 넘어가면 "아무것도 수집 안 됨"과 구분이 안 되기 때문
  - `match(spec, names=None)`, `save()`, `load(path)` (`__collectors.json`에 name→클래스 ref + path)
  - 여러 실행이 한 레지스트리를 공유하면 메트릭이 한곳에 모여 비교 가능

- **Collector** (`_base.py`): 기본 클래스
  - `__init__(name, connector)`, `path`는 `Collectors.set_collector` 시 설정
  - 라이프사이클: `_start(node)`, `_collect(node, idx, inner_idx, context)`, `_end_idx(node, idx)`, `_end(node)`
  - 에러 처리: `_collect`/`_end_idx`는 safe wrapper로 try/except 래핑; `_start`/`_end`는 직접 호출 — 에러 시 `warnings` 리스트에 저장 후 warning 로그
  - `on_attach(experimenter)`: `exp()`가 호출 — experimenter identity 비교로 중복 재계산 방지; `_on_attach(experimenter)` no-op 훅을 subclass에서 override
  - `_experimenter`: pickle 제외 (save/load 시 None으로 초기화)
  - `has_node(node)`: 수집 결과 보유 여부 (구 `has()`는 중복이라 제거됨)
  - `reset_nodes(nodes)`(base: `self._buf`에서 해당 노드 제거 — 서브클래스는 `super().reset_nodes(nodes)` 먼저 호출 후 자신의 disk/cache 정리), `save()`, `load(cls, path)`
  - `_get_nodes(nodes, available)`: None/list/str(regex) 패턴 매칭
  - context: `{node_spec, processor, info, input, outer_idx, inner_idx, output_train, output_valid, output_test, output_ext}` — 2026-08-02에 `node_attrs`→`node_spec`(ProcessorSpec), `spec`→`info`로 개명(예전 `spec` 키가 담던 건 `_process()`의 info dict라 새 `ProcessorSpec`과 이름이 겹쳤음)

- **MetricCollector** (`_metric.py`): 메트릭 수집
  - `output_var`(DSL 문자열 또는 None), `metric_func`, `include_train`
  - target: `context['input']['y']`, 예측값: `eval_expr(parse(output_var), output_valid, processor=context['processor'])`로 컬럼 선택
  - `_on_attach`: `metric_func`에 `on_attach`가 있으면 자동 전파
  - 저장: `push()` 오버라이드로 inner fold 결과 발생 시 즉시 `metrics.db` INSERT (per-node pkl 없음)
  - 쿼리: `get_metric(node)`, `get_metrics(nodes)`, `get_metrics_agg(nodes, inner_fold, outer_fold, include_std)`

- **ProbToLabel** (`_metric.py`): predict_proba → label 변환 후 metric 적용
  - `__init__(metric_func, var, thresholds=None)` — `metric_func`를 래핑하는 callable class
  - `var`: DSL 문자열 (예: `'{target}'`) — `on_attach`에서 `experimenter.get_test_data({'_y': var})`로 resolve
  - `thresholds`: None=argmax, float=binary threshold, list=multiclass per-class threshold
  - `on_attach`에서 experimenter로부터 label classes 추출 (정렬 순서 = predict_proba 열 순서)
  - binary: 2D proba `(n, 2)` 자동 처리 (col 1 추출), 1D sigmoid도 지원
  - multiclass per-class threshold: threshold 초과 클래스 중 최대 확률 선택, 없으면 argmax fallback

- **StackingCollector** (`_stacking.py`): 스태킹 데이터 수집
  - `__init__(name, connector, output_var, method='mean')` — experimenter 불필요
  - `_on_attach`에서 experimenter로부터 `_index`, `_target`(ndarray), `_target_columns`, `_data_cls` 구축
  - `output_var`, `method`(mean/mode/simple)
  - `_aggregate()`: `DataWrapper` 대신 `_data_cls`(입력 데이터 타입)의 static 메서드 사용
  - 쿼리: `get_dataset(nodes=None, include_target=True)`

- **ModelAttrCollector** (`_model_attr.py`): 모델 속성 수집 (feature_importances 등)
  - `result_key`, `adapter`(default=None, `get_adapter(connector.processor)`로 자동 설정), `params`
  - `_is_mergeable()`: self.adapter에서 직접 판단
  - 쿼리: `get_attr(node, idx)`, `get_attrs(nodes)`, `get_attrs_agg(node, agg_inner, agg_outer)`

- **SHAPCollector** (`_shap.py`): SHAP value 수집 및 분석
  - `explainer_cls`(default=shap.TreeExplainer), `data_filter`(DataFilter 인스턴스)
  - train/valid 각각 필터 적용 → SHAP 계산 → raw output 저장
  - 결과: `results[node][(idx, inner_idx)] = {'train', 'valid', 'train_index', 'valid_index', 'columns'}`
  - 분석: `get_feature_importance(node, idx)` → inner fold별 `pd.Series` 리스트
  - 분석: `get_feature_importance_agg(node, agg_inner='mean', agg_outer='mean')` → agg_inner=None이면 MultiIndex, agg_outer=None이면 DataFrame, 둘 다 설정이면 Series
  - SHAP 3D array(multiclass) 지원: `(n_samples, n_features, n_classes)` → class축 평균 후 처리

- **OutputCollector** (`_output.py`): output_train/output_valid 원본 저장
  - `output_var`, `include_target`
  - 파일 저장: `{path}/{node}/{idx}_{inner_idx}.pkl`
  - 쿼리: `get_output(node, idx, inner_idx)`, `get_outputs(node)`

- **ProcessCollector** (`_process.py`): 외부(테스트) 데이터에 대한 예측 수집
  - `__init__(name, connector, ext_data, output_var=None, method='mean')`
  - `collect`: `context['output_ext']`에서 결과 추출 → `output_var`로 컬럼 필터
  - inner fold 결과는 `method`(mean/mode/simple)로 outer fold별 집계, 파일 저장: `{path}/{node}/{idx}.pkl`
  - 쿼리: `get_output(nodes=None, agg='mean')` — nodes 필터(None/list/regex) + outer fold 집계 후 column-wise concat 반환
  - save/load 시 `ext_data`는 미저장 (런타임 전달)

## edges 구조 — DSL 문자열 (`_edge_dsl.py`)
- dict 형태: `{key: dsl_string}` — key는 변수 집합 이름(예: 'X', 'y', 'sample_weight'), 값은 **항상 순수 문자열**
- `edges[key]`는 정의/상속/직렬화/비교 어디서나 문자열 그대로 유지되며, **Processor 실행 시점에 실제 데이터를 대상으로만** lazily 컬럼 리스트로 확장됨(`eval_expr`, `_flow.py.get_data`에서 호출). `set_grp`/`set_node`는 DSL *구조*(문법 + namespace 참조)만 검증(`validate_edges`)하고 컬럼/schema는 절대 만들어보지 않음
- 문법:
  ```
  expr        := term (op term)*            -- op는 '+'/'-'/'&', 공백으로 양쪽이 분리된 독립 토큰이어야 함
  term        := ('*' | set_literal | pattern) ['@' NAME ['(' ')']]
                  | slice | namespace | '(' expr ')'
  slice       := [INT] ':' [INT]             -- 파이썬 slice, 예: '-1:' == slice(-1, None)
  set_literal := '{' [NAME (',' NAME)*] '}'  -- 명시적 컬럼명 목록
  namespace   := NAME ':' '(' expr ')'       -- NAME은 stage 노드명, 그 노드의 출력을 가리킴
  pattern     := REGEX                       -- re.match 로 컬럼명 매칭
  ```
  - `*`/`set_literal`/`pattern` 뒤에 바로(공백 없이) `@name`/`@name()`을 붙이면 `col.py`에 등록된 column-selector 적용 (예: `*@numeric`, `{a, b}@int`, `A.*@ohe_drop_first`)
  - 값을 직접 명시하지 않은 top-level(namespace 밖) 항목은 DataSource를 가리킴; `name:(...)` 블록은 그 stage 노드의 출력을 가리킴 (namespace는 top-level에서 `+`로만 결합 가능 — `-`/`&`는 namespace 내부/괄호 안에서만)
  - **DataSource 참조는 반드시 명시적 컬럼명 리스트(`{a, b}`)** — 패턴/callable 아님. [[feedback_datasource_edges_explicit_vars]]
- 그룹/노드 상속: 자기 값이 `+`/`-`로 시작하면 부모의 이미 resolve된 문자열에 이어붙임(`f"{parent} {own}"`); 그 외 일반 문자열은 완전히 override(상속 안 함). 자기 값이 없으면(`{}`) 부모 값을 그대로 상속
- 같은 key의 여러 segment(`+`로 연결)는 column 방향으로 concat됨

## Edge DSL 관련 함수 (`_edge_dsl.py`)
- `parse(dsl_string)` → AST (`Star`/`SetLiteral`/`Pattern`/`Namespace`/`BinOp`/`slice`)
- `eval_expr(node, data, processor=None)`: AST를 실제 `data`(`DataWrapper`, `get_columns()`/`select_by_dtype()` 노출)에 대해 평가 → 컬럼명 리스트. `data`에서 `columns = data.get_columns()`를 내부적으로 유도하므로 호출부는 컬럼 리스트를 따로 넘기지 않음
- `validate_edges(dsl_string, pipeline)`: 구조만 검증(문법 + namespace가 존재하는 노드를 가리키는지) — 컬럼/schema는 절대 건드리지 않음. Pipeline이 stage 전용이 되면서 role 검사는 사라짐
- `iter_segments(dsl_string)` → `(node_name, expr)` 이터레이터 (top-level `+` 체인 분해)
- `referenced_nodes(dsl_string)` → 참조하는 노드명 집합 (`None`=DataSource 포함)
- `unparse(node)` → AST를 다시 DSL 문자열로 렌더링

## col.py — `@name` column-selector 레지스트리
- `col_selector(*processor_classes, name=None)` 데코레이터로 등록. 모든 selector는 동일 시그니처 `(data, processor=None) -> mask` — `data`는 이미 패턴 등으로 좁혀진 후보 컬럼만 담은 `DataWrapper`
- `processor_classes`가 지정되면 해당 processor 타입에서만 유효(`resolve_selector`가 불일치 시 ValueError); 비워두면(`()`) processor 없이도 사용 가능
- `name=`으로 등록 키를 함수명과 다르게 지정 가능 (파이썬 builtin과 겹치는 `float`/`int`/`string` 등에 사용)
- **`ohe_drop_first`** (OneHotEncoder 전용), **`subset_poly`** (PolynomialFeatures 전용, degree/interaction/bias 일관된 전체 조합으로 스냅)
- **dtype 기반 builtin selector** (processor 불필요, `data.select_by_dtype(kind)` 사용): `@numeric`, `@categorical`, `@binary`(bool dtype만), `@float`, `@int`, `@string`
  - DataSource 최상위(`*@numeric` 등)에 바로 걸면 schema에 없는 raw 컬럼(id/target/sample_weight 등)까지 포함될 수 있어 위험 — 이미 확정된 stage 노드 출력 namespace 안에서 쓰는 게 안전

## exist 파라미터 (set_grp, set_node, set_collector)
- `'diff'` (default, set_grp/set_node): 제공된 파라미터가 기존과 다를 때만 업데이트, 동일하면 skip
- `'skip'` (set_collector default): 이미 존재하면 무시하고 반환
- `'error'`: 이미 존재하면 에러
- `'replace'`: 기존 객체를 무조건 업데이트

### set_grp 업데이트 동작 (중요)
`exist='diff'`에서 변경이 감지되면 **제공된 모든 값으로 전체 필드를 대입**한다.
`None`/빈 값은 그대로 `None`/`{}`으로 덮어쓰므로, **유지하려는 필드도 반드시 명시**해야 한다.
```python
# 잘못된 예 — processor/edges/method가 None으로 덮어써짐
p.set_grp('scale', params={'with_std': False})

# 올바른 예
p.set_grp('scale', processor='sklearn.preprocessing.StandardScaler',
          method='transform', edges={'X': '{' + ', '.join(cols) + '}'},
          params={'with_std': False})
```
[[feedback_pipeline_direct_reference]]: `p`가 스코프에 있으면 `p.set_grp`/`p.set_node`를 직접 호출 (`e.pipeline.set_grp`처럼 우회하지 않음)

## Staleness — 두 Pipeline 버전을 비교한다
serial 비교가 아니라 **버전 간 구조 비교**로 판정한다. 판정 지점은 `set_pipeline()` **한 곳**이고, `build()`/`train()`은 "디스크에 있는 건 유효하다"를 전제할 수 있다.

**`Pipeline.diff_from(old)`** — DataSource에서 위상 순서로 내려가며:
1. 이름이 old에 없으면 → stale
2. 정의(`_definition_of` = `processor/method/adapter/params/edges`)가 다르면 → stale
3. 정의는 같아도 **읽는 노드 중 하나가 stale이면** → stale (하위 전파가 위상 순서에서 자동으로 나옴)
4. old에는 있는데 지금 없는 이름 → stale (아티팩트 청소용)
5. DataSource schema/targets가 바뀌면 → 전부 stale

- **Trial 처리(2026-08-01 개정)**: Trial은 Pipeline 밖이라 diff가 이름을 모르고, stale stage를 읽은 Trial도 이제 안 건드림 — `set_pipeline`은 stale stage만 `reset_nodes`로 지우고 끝. Trial 아티팩트가 어떤 pipeline 버전에서 만들어졌는지는 `TrialStore.experiment_hist`가 기록해두므로 그대로 historical record로 남고, 다시 돌리려면 명시적으로 재실행해야 함(자동 캐스케이드 없음)
- **Trial 자신의 재정의 감지도 안 함(2026-08-01 제거)**: 예전엔 `_make_jobs`가 `info['definition'] != _definition_of(attrs)` 값 비교로 아티팩트가 지금 정의와 다르면 자동 reset 후 재실행했는데, 이 비교 자체를 없앰 — 지금 `Experimenter._make_jobs`는 오직 `TrialStore.experiment_hist`의 fold별 status만 봄: `'built'`면 스킵, `'error'`거나 기록이 없으면 job 생성. 재정의된 trial이 이미 `'built'`로 기록돼 있으면 조용히 스킵되므로, 다시 돌리려면 `reset_nodes([trial_name])`을 직접 호출해야 함
- **`serial` 자체가 없음(2026-08-01 전체 제거)**: 예전엔 정의 변경마다 새 UUID를 부여해 "뭔가 바뀌었다"는 신호로 썼지만, "결과가 달라지나"는 말 못 했다 — 지금은 `diff_from`이 정의를 직접 비교하므로 그 신호 자체가 불필요해짐. 이게 가능해진 건: **자기가 읽지 않는 stage를 고쳐도 Trial이 유지된다**(예전 serial은 전역이라 이 구분이 안 됐음)는 성질이 부산물로 따라옴

## Lazy resolution: processor / adapter / params (edges DSL과 동일한 지연 원칙)
- `set_grp`/`set_node`는 `processor`/`adapter`/`params`를 **스펙 형태만 검증하고 절대 resolve/instantiate하지 않음** — 실제 값으로의 변환은 전부 사용 시점(`_node_processor.py`)으로 미룸
- **산 객체는 정의 시점에 `TypeError`로 거부됨** (`_validate_processor`/`_validate_adapter`/`_validate_params`, `_pipeline.py`). 에러 메시지가 써야 할 ref 형태를 그대로 안내함
- **processor**: `"module.ClassName"` **문자열만** (클래스 객체 거부). 실제 클래스로 resolve되는 유일한 지점은 `_node_processor.py`(`TransformProcessor`/`PredictProcessor.__init__`의 `resolve_processor()` 호출). `Connector`, `_describer.py`, `Experimenter._make_stage_jobs`/`Trainer._make_stage_jobs`(GPU 판정)는 전부 이 문자열을 그대로 다룸
  - **클래스를 허용하면 안 되는 이유**: `Connector.match`는 `spec.processor`를 **문자열 그대로 비교**하므로, 클래스로 정의된 노드는 문자열 ref로 설정한 Connector와 **영영 매칭되지 않음** — 에러 없이 collector가 조용히 아무것도 수집하지 않음. `serialize_value`가 클래스를 `{"__type__":"class"}`로 저장하고 리로드 시 클래스로 되돌리므로 재시작해도 해소되지 않음
- **adapter**: `None` / `"module.ClassName"` 문자열 / `{"__ref__":...,"__params__":{...}}` dict — 인스턴스 거부. `resolve_node_adapter(processor, adapter_spec)`(`adapter/__init__.py`)가 사용 시점에 resolve
- **params**: 순수 데이터(스칼라/numpy 스칼라/list/tuple/dict)와 ref spec만 허용 — `{'__ref__':...}`(예: ColSelector, `mllab_sampler`)/`{'__callable__':...}`(예: metric 함수). 중첩 값까지 재귀 검증하며 에러가 경로(`params['a']['b'][0]`)를 표시함. `_node_processor.py`의 `_resolve_params()`가 Processor 생성 시점에 `resolve_ref_values()`로 해제
- **왜**: (1) `mllabs.nn.NNClassifier`처럼 무거운 import(TensorFlow)를 유발하는 processor/adapter가 파이프라인 "정의" 시점에 실수로 로드되는 걸 방지 — 실제 `build`/`exp` 실행 시점까지 미뤄짐. (2) 파이프라인 전체가 직렬화 가능해짐(declarative config 방향). (3) diff 비교가 raw spec 비교라 `_params_equal`이 `==` 한 줄로 축소됨 — 인스턴스 `__eq__` 신뢰성 이슈 자체가 사라짐
- **테스트**: `tests/mock.py`에 여러 테스트 파일에 흩어져 있던 더미 processor 클래스 중 string-ref로 참조돼야 하는 것들을 모아둠 — `tests/`엔 `__init__.py`가 없어 pytest가 bare module(`import mock`)로 수집하므로 `processor='mock.DummyStage'`식으로 참조 가능

## Processor (`_node_processor.py`)
- **TransformProcessor**: `fit`, `fit_process`, `process`
- **PredictProcessor**: `fit`, `fit_process`, `process`
- **processor(=transformer/estimator)/adapter/params가 실제 값으로 resolve되는 유일한 지점** — Pipeline은 이 셋을 절대 resolve 안 하고 spec 그대로 넘김(아래 "Lazy resolution" 섹션 참조)
  - `__init__`: `self.transformer`/`self.estimator = resolve_processor(transformer/estimator)`(`"module.ClassName"` 문자열 → 클래스, `_serialize.py`) — 이 라인이 processor가 클래스가 되는 유일한 곳
  - `self.adapter = resolve_node_adapter(transformer/estimator, adapter)` — **resolve 전의 raw(문자열) processor**를 넘김(resolve된 클래스가 아님 — `get_adapter`가 문자열/클래스 둘 다 처리하므로 순서 무관하지만, "processor는 인스턴스 생성 시점에만 클래스로" 원칙에 맞춰 문자열째로 전달)
  - `self.params = _resolve_params(params)` — `params` 내 `{'__ref__':...}`/`{'__callable__':...}` 항목을 `resolve_ref_values()`로 해제(ColSelector 인스턴스화 등). `mllab_sampler` 값도 여기서 같이 resolve됨
- `fit`/`fit_process`에서 y 데이터를 `squeeze()` 후 전달 (sklearn DataConversionWarning 억제)
- `get_feature_names_out` 반환값은 `list()` 로 변환하여 사용 (list/ndarray 호환)
- `process()`: `adapter.get_process_data(data)` 로 입력 타입 변환 — polars 등 라이브러리별 호환성 처리
- `data_dict` (Experimenter): `{key: ((train, train_v), valid), ...}` 형태
- `data_dict` (Trainer): `{key: (train, valid), ...}` 형태 (inner fold 없음)
- **X-less 지원**: `edges`에 `'X'`가 없고 `'y'`만 있는 경우(e.g. `LabelEncoder`) `'y'`를 primary input으로 사용
  - `fit`/`fit_process`: `'X'` 없으면 `'y'` 데이터를 squeeze하여 전달, `output_vars`를 `y_columns`로 설정
  - `process`: `X_`가 비어 있으면 입력 데이터를 squeeze 후 transform
- `y_columns`가 str인 경우(polars Series 등) `[y_columns]` 로 wrap하여 처리

## Adapter 인터페이스
- `get_params(params, logger)`: 모델 생성 파라미터
- `get_fit_params(data_dict, params, logger)`: fit 파라미터 — base: X/y를 `unwrap()` 후 반환
- `get_process_data(data)`: `process()` 입력 데이터 변환 — base: `unwrap(data)`
  - `LightGBMAdapter`: polars→pandas 변환 (LightGBM polars 미지원); `early_stopping` dict 수락 → 내부에서 `lgb_early_stopping` 콜백으로 변환 (params에 콜백 인스턴스를 넣을 수 없으므로 이 dict 형태가 유일한 지정 방법)
  - `CatBoostAdapter`: `_catboost_supports_polars()` (>=1.3.0) 기반 분기 — 구버전이면 polars→pandas (`get_fit_params`도 동일 적용)
- `result_objs`: `{name: (callable, mergeable_bool)}`
- `__eq__`: `type(self) is type(other) and self.__dict__ == other.__dict__`
- `__hash__`: `id(self)` — set/dict 키로 사용 가능
- **adapter 지정 방식** (`set_grp`/`set_node`의 `adapter=`): `"module.ClassName"` 문자열 / `{"__ref__": ..., "__params__": {...}}` / `None`만 허용(**인스턴스는 `TypeError`**) — **저장 시점엔 resolve 안 함**, `_node_processor.py`가 인스턴스 생성 시 `resolve_node_adapter(processor, adapter)`로 resolve(`adapter.resolve_node_adapter`, `adapter/__init__.py`)
  - `resolve_node_adapter(processor, adapter_spec)`: `adapter_spec` 있으면 `resolve_instance(adapter_spec)`, 없으면 `get_adapter(processor)`(processor 클래스명 기반 디폴트 — `get_adapter`는 문자열이면 `rpartition('.')[-1]`로 bare/`"module.ClassName"` 둘 다 처리, 클래스/인스턴스면 `.__name__`/`.__class__.__name__`)
  - GPU 판정(`need_gpu`)도 이 함수로 resolve — Trial은 `_make_jobs`가, Stage는 `_make_stage_jobs`가 job 생성 시점에 노드/trial 이름당 1회 resolve해 `Job.need_gpu`에 박아 넣음(2026-08-01부터 executor 쪽엔 GPU 판정 캐시가 없음 — job 리스트 자체가 이미 분류돼 있음)
- **레지스트리** (`adapter/__init__.py`): `MODEL_ADAPTERS`(모델명→인스턴스), `get_adapter(model_or_name)`. `NNAdapter`는 TF를 top-level import하므로 **지연 로드** — `_LAZY_ADAPTERS`(`NNClassifier`/`NNRegressor`)로 first-use 시 인스턴스화·캐시, 모듈 `__getattr__`로 `NNAdapter` 심볼 노출 → `import mllabs`가 TF를 끌어오지 않음

## Sampler (`sampler/` 패키지)
- **Sampler** (`_base.py`): 기본 클래스 — `sample(fit_params) → fit_params` 인터페이스
- **ImbLearnSampler** (`_imblearn.py`): imblearn `fit_resample` 래퍼
  - `__init__(sampler)`: imblearn sampler 인스턴스 주입
  - `sample(fit_params)`: `fit_params['X']`/`['y']`로 `fit_resample` 호출 후 X, y 교체하여 반환
- 사용법: node `params`에 `mllab_sampler` 키로 Sampler 인스턴스 지정 → `_node_processor`가 fit/fit_process 전에 `sample()` 호출; estimator에 전달 전 키 제거

## 보조 모듈
- **_data_wrapper.py**: DataWrapper (wrap/unwrap/squeeze/mean/mode/simple) — pandas/polars/cudf/numpy 통합
  - `PolarsWrapper.get_columns()`: `pl.DataFrame`이면 `.columns`, `pl.Series`이면 `.name` 반환
  - `select_by_dtype(kind)`: `'category'|'numeric'|'int'|'float'|'str'|'bool'`에 해당하는 컬럼명(numpy는 정수 offset) 리스트 반환 — `col.py`의 `@numeric` 등 dtype selector가 사용하는 primitive. (예전 `get_column_list(ColSelector(col_type=, pattern=))`는 제거됨 — pattern 부분은 이제 DSL의 `Pattern` 노드가 담당)
- **_edge_dsl.py**: edges DSL 파서/평가기 — 위 "Edge DSL" 섹션 참조
- **_serialize.py**: ref 기반 직렬화/해석
  - `serialize_value`/`deserialize_value` (JSON 왕복), `_obj_to_ref`/`_ref_to_obj`
  - `resolve_processor(x)`: `"module.ClassName"` str → 클래스, else passthrough
  - `resolve_instance(spec)`: str→인스턴스(기본값) / `{__ref__, __params__}`→`cls(**params)` / else passthrough. `resolve_adapter`가 위임
  - `resolve_ref_values(value)`: params 값 재귀 해석 — `{"__callable__": "mod.fn"}`→**호출 안 하고** 그 객체 참조(metric_func 등), `{"__ref__": ..., "__params__": {...}}`→인스턴스화, 문자열/스칼라는 그대로. `set_grp`/`set_node`/`set_collector`의 params에 적용
- **_experimenter_store.py**: `ExperimenterStore` — 프로젝트 전역 `experimenters.db`, `name`이 PK인 타입 있는 컬럼
- **_project.py / _trial.py / _trial_store.py**: 위 해당 섹션 참조
- **_executor.py**: 실제 실행
  - **`Job(name, spec, outer_idx, inner_idx, flow, need_gpu=False)`(2026-08-01, `StageJob`/`TrialJob` 통합)** — Stage와 Trial 공용 job 단위. `spec`은 `Pipeline.get_node_spec()`/`Trial.get_spec()`이 준 `ProcessorSpec`을 job 생성 시점에 1회 계산해 박아 넣은 것(따로 `node`/`trial` 객체를 들고 있지 않음). `flow` 하나로 `get_train`/`get_valid`/`get_test(edges)`를 다 만들 수 있어서(`TrainDataFlow`) `outer_folds`/`train_folds` 참조가 필요 없어짐. `node_path()`는 `flow.node_path(name)`에 위임
  - **`_execute_single(jobs, store, gpu_id_list=None, collectors=None, tracker=None)`(2026-08-01, `_build_flow_single`/`_experiment_single` 통합)** — 단일 프로세스로 `Job` 리스트를 그대로 실행. Stage/Trial 차이는 `collectors`뿐이었음(Stage는 Collector가 없음) — **`collectors=None`이 Stage/build 경로**: 입력 준비가 `ext_data` 없이 `_stage_job_data(job)`로 끝나고 매치/실행도 스킵. `collectors=[]`(Trainer의 Trial 경로 — Trainer도 Collector가 없음)는 실제 리스트와 같은 코드 경로를 타되 매치될 게 없을 뿐이라 결과는 동일 — `None`은 그 스킵만큼만 다름
    - 의존성 순서 `while True: ready = [...]` 루프(Stage끼리 서로 참조 가능해서 필요)는 Trial에도 그대로 적용되지만, 그 안의 `get_missing_nodes` 게이트는 **Stage 전용**(`collectors is None`일 때만 검사) — Trial의 edges는 항상 이미 빌드된 Stage만 참조하므로 보통은 검사해도 즉시 빈 리스트가 되지만, Trial까지 이 게이트에 걸리게 하면 참조하는 Stage가 끝내 안 빌드된 경우 그 Trial job이 (에러 하나 안 남기고) 조용히 영원히 대기만 하다 사라짐 — 원래 `_job_inputs`가 `TrainDataFlow._resolve_typ`에서 `KeyError`를 내고 그게 prep error로 잡혀 기록되던 것과 다른 동작이 되어버림(2026-08-01, `_execute_single` 첫 병합 때 들어간 버그, 이번에 `_execute_multi`와 함께 수정)
    - `job.flow.set_objs`는 Stage/Trial 구분 없이 완료된 job마다 호출됨 — 안 하면 그 job이 `ready`에서 절대 빠지지 않아(다른 무엇도 "완료"로 표시 안 하므로) 루프가 영원히 재실행함
    - `store`(그 run의 `NodeStore`)를 명시적으로 받아 `store.write_objs(node_name, outer_idx, inner_idx, obj, result)`를 호출 — `NodeStore`를 import해서 정적으로 부르던 방식(2026-08-01 이전)을 대체
    - 내부 `errors`는 항상 `(outer_idx, inner_idx, name)`로 키잉(실패한 job이 ready-루프에서 영원히 재시도되는 걸 막으려면 fold까지 포함한 식별자가 필요) — 반환값은 기존 호출부 계약에 맞춰 분기: `collectors=None`(Stage)은 그대로, 아니면(Trial) `(outer_idx, name)`로 축약해 `_execute_multi`와 모양을 맞춤(같은 outer fold의 다른 inner fold 에러를 덮어쓸 수 있음 — 이 병합 이전부터 있던 충돌이라 여기서 고치지 않음)
  - **`_execute_multi(jobs, n_jobs, store, gpu_id_list=None, collectors=None, tracker=None, ...)`(2026-08-01, `_build_flow_multi`/`_experiment_multi` 통합)** — 워커 풀로 `Job` 리스트를 실행. `_execute_single`과 같은 `collectors` 분기(`None`=Stage/build, 리스트=Trial/experiment — Trainer의 Trial 경로는 Collector가 없어도 `[]`를 명시로 넘김, 반환 키 축약 분기를 타야 해서)
    - **ready-job 계산은 매 dispatch 사이클마다 처음부터 다시 스캔**(`_collect_ready()`, 옛 `_build_flow_multi` 방식) — 옛 `_experiment_multi`처럼 `gpu_jobs`/`cpu_jobs` 두 리스트를 한 번만 만들어두고 dispatch/에러 때마다 지워나가는 방식(`_drop` 헬퍼, 이번에 제거됨)은 Stage에 안 맞음 — Stage는 형제 노드가 끝나야 readiness가 바뀌기 때문. 재스캔 방식은 Trial에도 그대로 맞음(Trial끼리는 서로 의존 안 하니 한 번 ready면 계속 ready). `get_missing_nodes` 게이트는 여기서도 Stage 전용(위 `_execute_single`과 같은 이유)
    - **워커 배정 fallback 정책은 옛 `_experiment_multi` 쪽을 채택**(Stage에도 동일 적용) — "내 타입" job이 아직 남아있으면 그 타입 몫 worker를 다른 타입에 안 뺏기는 정책(`elif free_cpu and not cpu_ready and gpu_fallback_cpu`)이 옛 `_build_flow_multi`의 무조건 fallback보다 나음. 단, ready 목록을 매 사이클 재계산하는 탓에 같은 `_try_dispatch()` 호출 안에서 GPU pass가 막 dispatch한 job이 CPU pass의 "내 타입 남았나" 판정에는 그 사이클 안에서 반영 안 됨(다음 'done'/'error' 이벤트에서 바로잡힘) — 무시할 만한 수준의 부정확
    - `ProcessWorker(conn, collectors or [], store, ...)`로 `store`를 그대로 넘김(2026-08-01) — 워커 메시지 튜플에서 `node_path`가 빠짐(`spec, outer_idx, inner_idx, train_data, valid_data, test_data, ext_data` — 워커가 `store.write_objs(spec.name, outer_idx, inner_idx, obj, result)`로 직접 이름/fold를 쓰므로 경로를 미리 조립해 보낼 필요가 없어짐)
  - `ProcessWorker`(spawn): job 경계에서 `del` + `gc.collect()`로 이전 job의 데이터·모델을 놓아줌(안 하면 피크 = 이전 데이터 + 모델 + 새 데이터). 워커 로그 fd는 dup2 직후 close
- **_tracker.py**: `ExecuteTracker` 기반
  - `LoggerExecuteTracker` — 워커 이벤트→logger. `typ`에 따라 `logger.info`/`warning` 라우팅
  - **`TrialHistTracker(tracker, store, experimenter, pipeline_version)`** — 로깅 tracker를 감싸 `done`/`error` 시점에 `TrialStore`에 이력 기록(`pipeline_version`은 그 Experimenter의 정수 버전, 해시 아님). 이벤트 시점이라 멀티워커도 그대로 커버되고, 사후에 디스크를 다시 읽지 않아도 됨
- **_describer.py**: desc_spec, desc_pipeline, desc_node, compare_nodes (`desc_status`는 죽은 코드라 제거됨)
- **_logger.py**: BaseLogger, DefaultLogger (start/update/end_progress, adhoc_progress, rename_progress)
- **col.py**: `@name` column-selector 레지스트리 — 위 "col.py" 섹션 참조
- **_connector.py**: Connector (노드 매칭)
- **collector/**: Collector, MetricCollector, StackingCollector, ModelAttrCollector, SHAPCollector, OutputCollector
- **filter/**: DataFilter, RandomFilter(n/frac/random_state), IndexFilter(index)
- **adapter/**: sklearn, xgboost, lightgbm, catboost, keras, `_nn.py` (NNAdapter)
- **processor/**: CatConverter, CatPairCombiner, CatOOVFilter, FrequencyEncoder, TypeConverter, CrossFitTransformer (`ColSelector`는 `_pipeline.py`에 있음 — processor/ 소속 아님)
  - `CatPairCombiner`: pair(2) → N-way 그룹 조합으로 확장. `pairs` 요소를 N개 컬럼 인덱스/이름 그룹으로 지정 가능
  - `TypeConverter`: 모든 컬럼을 지정 타입(`str`/`int`/`float`)으로 변환. pandas: `astype`, polars: cast, numpy: `astype`. `get_feature_names_out` 지원
  - `CrossFitTransformer`: sklearn-compatible stacking meta-feature 생성기
    - `__init__(estimator, cv=5, method='predict_proba', stratified=True)`
    - `fit_transform`: CV로 OOF 예측 생성 + 전체 데이터로 full estimator fit
    - `transform`: full estimator로 예측 (fit_transform 이후)
    - 출력 컬럼명: `{estimator_class_lower}_{class}` (predict_proba) / `{estimator_class_lower}_pred` (predict)
    - Stage 노드로 사용 시 Experimenter는 OOF, Trainer/Inferencer는 full model 경로로 동작
  - polars 설치 시: PolarsLoader, ExprProcessor, PandasConverter 추가
  - `_dproc.py`: `get_type_df` (수치형만 f32/i32/i16/i8 판정), `get_type_pl`, `get_type_pd`, `merge_type_df`

## 저장 구조
**Project가 경로를 소유한다.**
```
{project.path}/
  experimenters.db                  # experimenters (name PK, data_key, title,
                                    #                pipeline_name, pipeline_version)
  trials.db                         # trials + experiment_hist (status + info JSON)

  pipelines/{name}/
    {name}.db                       # PipelineBuilder 노드/그룹 정의 + versions(version PK, path) — 이 pipeline만의 버전 카운터
    v{n}.pkl                        # 버전별 빌드 결과 Pipeline (형식은 PipelineStore.save_version 뒤에 숨어 있음)

  collectors/
    __collectors.json               # name → 클래스 ref + path
    {name}/                         # Collector가 소유하는 저장 위치
      __config.pkl
      metrics.db                    # MetricCollector (node, idx, inner_idx, split, value)
      {node}.pkl                    # StackingCollector
      {node}/{idx}_{inner_idx}.pkl  # OutputCollector

  exp/{name}/                       # Experimenter — 이름이 곧 식별자
    __splitters.pkl                 # sp, sp_v, splitter_params (ref-직렬화 불가라 pickle)
    __worker_logs/worker_{i}.log    # 멀티워커가 캡처한 네이티브 출력 (+ master.log)
    __folds/__node_hist.db          # 이 run의 NodeStore가 소유 (2026-08-01, NodeInfoStore 통합)
    __folds/{outer_idx}/{inner_idx}/{name}/
      obj.pkl                       # processor 객체
      result.pkl                    # fit_transform/fit_predict 출력
      # info.pkl 없음(2026-08-01 제거) — status/definition/edges 등은
      # 전부 __node_hist.db(Stage)/../trials.db의 experiment_hist(Trial)에
      # 있음. stage와 trial이 같은 디렉토리를 쓰는 건 그대로고, 종류를
      # 나타내는 필드(role)도 이제 없음 — 굳이 구분해야 하면 그 두 history
      # 중 어디에 기록됐는지로 알 수 있음(NodeStore 자신은 obj.pkl 존재만
      # 앎 — status(name, outer_idx, inner_idx))

  trainers/{name}/
    __trainer.pkl                   # splitter, split_indices, (pipeline_name, pipeline_version)
    __node_hist.db                  # 이 run의 NodeStore 소유 — exp/{name}과 별개 base path라 겹칠 일 없음
    {split_idx}/{name}/             # obj.pkl / result.pkl (info 없음, 위와 동일)

  inferencers/{name}/
    __inferencer.pkl                # node_specs, selected_stages/heads, n_splits, node_objs, v
```
- Experimenter/Trainer 디렉토리에 **`pipeline.pkl` 사본이 없다** — 포인터만 저장하고 Pipeline은 프로젝트에 한 벌만 존재
- **NodeStore는 project 전역이 아니라 run(Experimenter/Trainer) 하나당 하나(2026-08-01)** — `project.node_info` 같은 프로젝트 전역 레지스트리 없음. `Experimenter.node_store`/`Trainer.node_store`가 생성자에서 자기 base path로 만들어 모든 fold가 공유

## 패키지 정보
- PyPI 패키지명: `ml-labs`, Python 패키지: `mllabs/`
- `pyproject.toml`: setuptools 기반, Python >=3.10
- optional deps: `xgboost`, `lightgbm`, `catboost`, `shap`, `polars`, `tensorflow`, `all`, `dev`
- 릴리즈: `v*` 태그 push → GitHub Actions (`publish.yml`) → 테스트(3.10/3.11/3.12) → build → PyPI 자동 배포 (OIDC)

## mllabs.nn 패키지
- `NNClassifier`, `NNRegressor`: sklearn-compatible TF/Keras 기반 추정기
  - pandas `Categorical` / polars `Categorical`/`Enum` dtype 자동 감지 → embedding 자동 생성
  - `embedding_dims`: `{col: dim}` dict로 per-column override
  - `head`: head factory 클래스 (default=`SimpleConcatHead`), `head_params`: head factory에 전달할 kwargs dict
  - `hidden`: `DenseHidden` 인스턴스 또는 dict (kwargs로 전달) 또는 None(기본값)
  - `fit(X, y, eval_set=None, callbacks=None)`: constructor callbacks + fit callbacks + early stopping 순서로 합산
  - `evals_result_`: `{'train': {metric: [...]}, 'valid': {metric: [...]}}` (history 저장)
  - Pickle: `__getstate__`/`__setstate__` — weights만 저장, `col_info_` 기반 architecture 재빌드
- 컴포넌트: `SimpleConcatHead`, `FTTransformerHead`, `DenseHidden`, `LogitOutput`, `BinaryLogitOutput`, `RegressionOutput`
  - `FTTransformerHead`: Feature Tokenizer + Transformer head
    - cat embedding → d_model projection, cont feature → per-feature learned (w, b) tokenization
    - CLS token prepend + N × FTBlock (pre-LN, MHA + FFN/GELU, residual dropout) → CLS token 반환
    - 파라미터: `d_model=192`, `n_heads=8`, `n_layers=3`, `ffn_factor=4/3`, `attention_dropout=0.2`, `ffn_dropout=0.1`, `residual_dropout=0.0`
- `NNAdapter` (`adapter/_nn.py`): eval_set 전달 + `_ProgressCallback` (epoch 진행률 로깅) + `evals_result` result_obj
