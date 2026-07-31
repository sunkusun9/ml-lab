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
- **Project** (`_project.py`): 디렉토리 레이아웃 소유 + `TrialStore`/`ExperimenterStore` 레지스트리. Pipeline 버전은 Project가 색인하지 않음 — 각 pipeline이 자기 db에 자기 버전을 직접 관리(`build_pipeline` 참조). 컴포넌트는 여전히 단독 동작 가능하지만 Experimenter/Trainer는 Project를 요구
- **PipelineBuilder / Pipeline** (`_pipeline.py`): 가변 빌더 + `build()`가 만드는 불변 **stage 전용** 그래프
- **Trial / make_trials** (`_trial.py`): 평가할 구성 하나 = 예전의 Head 노드. Pipeline 밖에 있음
- **TrialStore** (`_trial_store.py`): `trials`(정의) + `experiment_hist`(fold별 실행 이력)
- **Experimenter** (`_experimenter.py`): CV 실험 실행/관리
- **Trainer** (`_trainer.py`): 학습 실행/관리 (split 기반)
- **Inferencer** (`_inferencer.py`): 학습된 processor를 새 데이터에 적용
- **NodeStore** (`_store.py`): 노드 아티팩트 읽기/쓰기 (obj.pkl / result.pkl / info.pkl)
- **DataFlow / TrainDataFlow** (`_flow.py`): fold별 데이터 흐름 및 stage 빌드
- **_executor.py**: `_build_flow_single/multi`(stage), `_experiment_single/multi`(trial) — 실제 실행

## Node/Trial 상태 모델

### 4-State
`init → built → finalized` / `init → error → (reset) → init`

| 상태 | Disk | 설명 |
|------|------|------|
| **init** | - | 정의만 된 상태 |
| **built** | O | 빌드 완료, 결과 추출 가능 |
| **finalized** | info only | obj/result 삭제 (`close_exp` 경로) |
| **error** | info only | 실행 중 에러 발생, 내역 보존 |

- `exp()`에 finalize 옵션은 없음 — `close_exp()`만 finalized로 보냄

### Experimenter 2-State
`open → closed`
- **open**: 정상 동작
- **closed**: `close_exp()` 호출 → 빌드된 노드를 일괄 finalize. Collector 데이터는 잔존

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
- **Pipeline은 stage 전용** — `role` 파라미터가 없음. `_BuiltNode.role`은 클래스 상수 `'stage'`로, Connector가 Trial의 `'head'`와 구분하는 용도로만 attrs에 실림
- grp는 build를 넘어가지 않음 — 원래 그룹명은 표시용 `label`로만 남음 (`get_node_attrs`의 키도 `'grp'` → `'label'`)
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
- `get_node_names(query)`, `get_node_attrs(name)`, `_find_descendants(name)`
- `sync()`: DB가 source of truth. 그룹/노드 필드를 직접 값 비교(`diff()`)해 갱신하고, **그룹이 바뀌면 그 그룹(+자식 그룹) 소속 노드들의 attrs 캐시도 함께 무효화**해 `changes['nodes']['updated']`에 포함시킴(노드 자신의 행은 안 바뀌었어도 상속받는 값이 바뀌었으므로)
- **`serial` 없음(2026-08-01 제거)**: 예전엔 정의 변경마다 새 UUID를 부여해 staleness/버전 판정에 썼지만, 지금은 두 용도 모두 다른 방식으로 대체됨 — staleness는 `Pipeline.diff_from`(아래)의 값 비교, 버전은 해시/dedup 없이 `PipelineStore`가 관리하는 단순 `max+1` 카운터(`Pipeline.content_key`도 없음 — 아래 "저장 구조" 참조)
- `copy()`, `copy_nodes(node_names)` — 선택적 복사 (builder→builder)
- `compare_nodes(nodes)` → `{processor_name: DataFrame}` (params 차이 + edges['X'] stage별 변수 차이)
- `desc_pipeline(max_depth, direction)`, `desc_node(node_name, direction, show_params)`: Mermaid 다이어그램 — grp 계층이 필요하므로 **builder 전용**

#### Pipeline (빌드 결과)
- `nodes`: `{name: _BuiltNode}` — `None` 키는 `_BuiltDataSource` (builder와 동일한 관례)
- `_BuiltNode` 속성(`__slots__`): `name`, `label`, `processor`, `edges`, `method`, `adapter`, `params`, `desc`, `output_edges` (+ 클래스 상수 `role='stage'`)
- `pipeline_id`(builder 신원) / `build_id`(빌드 호출마다 새 UUID) / `version`(`int | None`) — **`Project.build_pipeline()`이 저장할 때만 세팅**. `builder.build()`를 직접 부르면 `None`(미저장 in-memory 빌드)
- `get_node(name)`, `get_node_attrs(name)`, `get_node_names(query=None)`
- `topo_order()`: DataSource에서 내려오는 깊이순 노드명 (DataSource 제외) — 빌드 시 1회 계산해 캐시
- `descendants(name)`, `check_data_compatibility(data)`
- **`diff_from(old)`** → `set[str]`: 아래 "staleness" 섹션 참조
- `subset(node_names)`: 지정 노드 + 조상만 담은 새 Pipeline
- **불변성의 한계**: `params`/`edges`는 shallow copy — 중첩 값은 builder와 공유. "수정하지 않는다"는 관례로 지킴

- **`_DataSourceNode`** (`_PipelineNode` 서브클래스):
  - `schema`: `{col: var_type}` — var_type은 VAR_TYPES 중 하나
  - `targets`: `list[str]` — 타겟 컬럼 목록 (타입과 별도)
  - `get_attrs(grps)`: role='datasource', schema, targets 반환 (processor/edges/method/params 없음)

- **`_PipelineGroup`**: 노드 그룹 — builder 내부 전용
  - 속성: `name`, `processor`, `edges`, `method`, `parent`, `adapter`, `params`, `desc`
  - `children`: 자식 그룹명 리스트, `nodes`: 소속 노드명 리스트
  - `get_attrs(grps)`: 상위 그룹 속성 병합하여 반환 (`desc`는 상속 안 됨, 각 요소 독립)
  - `diff(processor, edges, method, parent, adapter, params)`: 달라진 필드명 리스트 반환 (`desc` 제외 → desc-only 변경은 rebuild 미유발)

- **`_PipelineNode`**: 개별 노드 — builder 내부 전용
  - 속성: `name`, `grp`, `processor`, `edges`, `method`, `adapter`, `params`, `desc`
  - `output_edges`: 이 노드를 입력으로 사용하는 노드명 리스트
  - `get_attrs(grps)`: 그룹 속성과 노드 속성 병합 (`role='stage'` 상수)
  - `diff(grp, processor, edges, method, adapter, params)`: 달라진 필드명 리스트 반환 (`desc` 제외)
  - `set_grp`/`set_node`: `desc` 파라미터 수락; exist='diff' skip 경로에서도 `desc`는 업데이트됨

- **ColSelector** (`_pipeline.py`): processor params(예: `cat_features`, `cat_cols`)에 쓰는 지연(lazy) 컬럼 선택자
  - `__init__(dsl_string='*')` — DSL 문자열 하나만 보유(정의 시점엔 데이터 불필요, `edges[key]`와 동일한 원칙)
  - **params에는 인스턴스가 아니라 ref-dict로 지정**: `{"__ref__": "mllabs.ColSelector", "__params__": {"dsl_string": "*@categorical"}}` (인스턴스는 `set_grp`/`set_node`가 `TypeError`로 거부)
  - `_node_processor`가 Processor 생성 시 `resolve_ref_values()`로 인스턴스화하고, fit 시점에 `_resolve_col_selectors`가 `eval_expr(parse(v.dsl_string), data)`로 컬럼 확정

### Trial (`_trial.py`)
Head를 Pipeline에서 떼어낸 결과. **Experiment 클래스는 없음** — Trial 리스트를 직접 넘긴다.

- **`Trial`**: 평가할 구성 하나. `name`, `processor`, `method`, `adapter`, `params`, `edges`, `label`, `tag`
  - `get_attrs()`: `Pipeline.get_node_attrs()`와 같은 모양(`role='head'` 고정) → Connector/executor/Collector가 stage 노드와 동일하게 취급
  - **이름이 식별자**. 디스크 아티팩트 디렉토리명이자 `TrialStore`(`trials` 테이블 PK)의 키 — 재정의하면 아티팩트도 `TrialStore` row도 덮어씀
  - `content_key()`: 정의(`processor/method/adapter/params/edges`)의 정규화 JSON 문자열. `name`/`label` 제외 — 이름을 바꿔도 계산 결과는 같음. **params가 순수 데이터로 강제된 덕에** 안정적 렌더링 가능. **어디에도 저장되지 않는 순수 값-비교 유틸** — 두 정의가 같은지 판별할 때만 씀
  - `stage_names()`: edges가 참조하는 stage 이름 집합

- **`make_trials(name, processor, edges, method, adapter, params, param_grid, tags)`** → `list[Trial]`
  - `params`(전 trial 공통) + `param_grid`(`{param: [values]}`) 카테시안 곱, grid 키 정렬 기준 결정적 순서
  - 이름: 단일이면 `{name}`, 복수면 `{name}_{idx}` (0 패딩)
  - `_validate_processor`/`_validate_adapter`/`_validate_params`로 spec 검증 (Pipeline과 동일 규칙)

### Project (`_project.py`)
디렉토리 레이아웃 소유 + 프로젝트 전역 레지스트리. 컴포넌트는 단독 동작 가능하지만 **Experimenter/Trainer는 Project를 요구**한다.

- `Project(path, cache_maxsize=4GB)` — `DataCache`를 소유하고 모든 Experimenter/Trainer가 공유
- 경로: `pipeline_path(name)`, `exp_path(name)`, `trainer_path(name)`, `inferencer_path(name)`, `collectors_path()`
- 팩토리: `pipeline_builder(name)`, `collectors()`, `experimenter(name, data, **kw)`, `load_experimenter(name, data)`, `trainer(name, data, **kw)`, `load_trainer(name, data)`
- **Pipeline 버전**: `build_pipeline(builder)` → `builder.build()` 호출 후 결과를 다음 버전(1부터, `builder._store`가 관리하는 카운터의 `max+1`)으로 저장하고 `pipeline.version`에 세팅해 반환. **content dedup 없음** — 내용이 같아도 호출할 때마다 새 버전(`builder`에 path가 없으면 `ValueError`)
  - 카운터/버전 파일은 **Project가 아니라 각 pipeline 자신의 db**(`pipelines/{name}/{name}.db`)가 소유 — `build_pipeline`은 `builder._store.save_version()`에 위임할 뿐, 프로젝트 전역 색인이 없음
  - `load_pipeline(name, version=None)`, `list_pipeline_versions(name)` — 둘 다 내부적으로 `PipelineStore(pipeline_path(name), name)`를 통해 조회
  - 저장은 pkl (`v{n}.pkl`) — 형식은 `PipelineStore.save_version`/`load_version` 뒤에 숨어 있어 나중에 교체 가능
- `trials`: `TrialStore`, `experimenters`: `ExperimenterStore`, `list_experimenters()`

### TrialStore (`_trial_store.py`)
```sql
trials(name PK, label, processor, method, adapter, params, edges, tag)
experiment_hist(trial_name, experimenter, outer_idx, inner_idx,  -- PK
                pipeline_version, status)
```
- **인조식별자도 content hash도 없음.** 두 테이블 다 **이름이 PK**(`trials`는 trial 이름 하나, 이력은 trial 이름 + experimenter 이름). `pipeline_version`은 해시가 아니라 **정수** — 그 실행의 `Experimenter.pipeline_version`을 그대로 기록
- 이름으로 키잉하는 이유: 아티팩트가 이미 이름으로 키잉돼 있음(`{exp}/__folds/{o}/{i}/{trial_name}/`, `{project}/exp/{name}`). 맞춰두면 조인 없이 읽히고, **정의를 바꿔 재실행 = 아티팩트 덮어쓰기 = 행 덮어쓰기**가 두 테이블 모두 일관됨(`register`는 `INSERT OR REPLACE`)
- **content_key 컬럼 없음(2026-08-01 제거)**: params가 평문 데이터로 강제된 덕에 정의 일치 여부는 값 비교 하나로 충분(`has()`), 아티팩트 rebuild 필요 여부도 디스크 `info['definition']` 값 비교로 이미 판정(`Experimenter._make_jobs`) — 해시 컬럼은 이 둘을 재서술할 뿐이었음. `experiment_hist`는 실행 로그일 뿐 정의의 출처가 아니라서, 이름이 재정의되면 예전 정의 자체를 복원하는 기능은 애초에 없음(`Trial.content_key()` 메소드 자체는 두 정의를 값으로 비교하는 유틸로 `_trial.py`에 남아 있으나 저장/식별용으로는 안 쓰임)
- `register(trial)`/`register_all(trials)`: 이름 기준 upsert(반환값 없음). `has(trial)`: 그 이름에 저장된 게 **지금** 이 정의와 같은지 필드별 비교. `get_by_name(name)`, `list_trials()`
- `record(trial_name, experimenter, outer_idx, inner_idx, pipeline_version, status)`, `get_hist(...)`, `get_status(...)`, `remove_hist(...)`

### Experimenter (`_experimenter.py`)
- 생성자: `Experimenter(project, name, data, ..., pipeline_name='pipeline', pipeline_version=None)` — 보통은 `project.experimenter(name, data, ...)`로 생성
- **이름이 식별자**: 경로는 `{project}/exp/{name}`, `TrialStore` 이력의 키도 이 이름. `exp_id` 같은 UUID 없음
- **Pipeline은 버전으로 지정** — `set_pipeline_version(version, pipeline_name=None)`이 Project에서 로드. `pipeline.pkl`을 실험 디렉토리에 복사하지 않고 **포인터(`pipeline_name`, `pipeline_version`)만** 저장
  - 버전 전환 시 `pipeline.diff_from(self.pipeline)`으로 stale 판정 → `_drop_stale()`이 해당 stage + 그걸 읽은 Trial 아티팩트를 제거
- `cache`: `project.cache` 공유 (크기는 `Project(cache_maxsize=)`에서 결정)
- `set_status(status)`: `self.status` 설정 + 프로젝트 `experimenters` 테이블의 status만 갱신. `open()`/`close()`/`close_exp()`/`reopen_exp()`가 사용
- **OS log capture** (`open_os_log`/`close_os_log`/`os_log`) — `open()`/`close()`(experiment status)와는 **무관한 별개 기능**:
  - `open_os_log(log_path=None)`: 이 프로세스의 OS-level stdout/stderr(fd 1/2)를 `{path}/__worker_logs/master.log`(기본값)로 dup2 리다이렉트 시작 — `self._os_log_state`에 원본 fd/`sys.stdout`·`stderr` 백업 보관. 이미 open이면 에러
  - `close_os_log()`: 리다이렉트 원복(`sys.stdout`/`stderr` 및 fd 1/2 복구). open 안 된 상태에서 호출하면 no-op
  - `os_log(log_path=None)`: 위 둘을 감싼 컨텍스트 매니저 — `with e.os_log(): e.build(n_jobs=1); e.exp(n_jobs=4)`
  - open~close 구간 동안: `n_jobs=1`인 `build`/`exp`는 같은 프로세스에서 돌기 때문에 마스터 리다이렉트가 그대로 캡처(별도 처리 불필요). `n_jobs>1`이면 그 구간에 한해 `log_dir`이 전달되어 워커별 리다이렉트도 같이 동작(위 `build`/`exp` 항목 참조)
  - `sys.stdout`/`stderr`는 원본 fd의 dup으로 rebind되므로, capture가 열려 있어도 `DefaultLogger`의 진행률 표시 등 Python 레벨 출력은 그대로 콘솔에 보임 — dup2로 fd 1/2만 로그 파일로 돌리기 때문에 native(C-level) 직접 write만 잡힘
- **pipeline 필요** (`_require_pipeline()`로 미설정 시 에러):
  - `build(nodes=None, rebuild=False, n_jobs=1, gpu_id_list=None, logger=None)` — stage 빌드
  - **`exp(trials, collectors=None, n_jobs=1, gpu_id_list=None, logger=None)`**
    - `trials`: **`[(Trial, outer_idx, inner_idx), ...]`** 튜플 리스트. fold 전개를 여기서 하므로 executor는 목록을 그대로 실행
    - `collectors`: `Collectors` 레지스트리 / Collector 인스턴스 리스트 / `None`
    - `finalize` 인자 없음
    - `_make_jobs()`가 `TrialJob(trial, attrs, cache_key, flow, need_gpu)` 리스트를 만듦. skip/reset 판정과 GPU 판정을 여기서 하고, adapter resolve는 **trial 이름당 1회**
    - Trial 정의를 `project.trials`에 등록하고, `TrialHistTracker`가 fold별 done/error를 이력에 기록
  - `n_jobs`는 실제 작업 수로 상한 처리 (`min(n_jobs, len(jobs))`) — 유휴 워커/progress bar 방지
  - `n_jobs > 1`이고 OS log capture가 open일 때만 워커 stdout/stderr를 `{path}/__worker_logs/worker_{i}.log`로 리다이렉트
  - `reopen_exp()`: closed→open, Stage 노드 초기화 후 `build()` 재호출
  - `get_node_info()`: 노드 요약 Markdown
- **pipeline 불필요** (디스크 상태만으로 동작): `get_status(node_name)`, `finalize(nodes)`, `reinitialize(nodes)`, `close_exp()`, `reset_nodes(nodes)`, `show_error_nodes(...)`, `get_objs(node_name, outer_idx=0, inner_idx=0)`
  - fold당 `NodeStore`가 **하나**(`train_data_flows[j]`)라 예전의 이중 store stale 캐시 문제는 없음
- **OS log capture** (`open_os_log`/`close_os_log`/`os_log`) — experiment status의 `open()`/`close()`와는 무관한 별개 기능:
  - `open_os_log(log_path=None)`: 이 프로세스의 fd 1/2를 `{path}/__worker_logs/master.log`로 dup2 리다이렉트
  - `close_os_log()`: 원복. `os_log()`는 둘을 감싼 컨텍스트 매니저
  - `sys.stdout`/`stderr`는 원본 fd의 dup으로 rebind되므로 진행률 등 Python 레벨 출력은 콘솔에 그대로 보임 — native(C-level) write만 잡힘
- `get_worker_logs(worker=None)`: 캡처된 네이티브 출력 — `{worker_idx: text, 'master': text}`. 매 실행마다 덮어씀
- `get_train_data(edges, o_idx=0, i_idx=0)` / `get_valid_data(...)` / `get_test_data(...)`: 출력 추출 헬퍼
- `aug_data`: 외부 데이터를 DataSource 수준에서 inner train split에 append — 미퍼시스트
- 저장/로드: `Experimenter.load(project, name, data, data_key=None)` (= `project.load_experimenter(...)`)
  - meta는 **프로젝트 전역 `experimenters.db`**에 (`_experimenter_store.py`) — `name`이 PK, 타입 있는 컬럼(`data_key, title, status, pipeline_name, pipeline_version`). 실험 디렉토리에 `__exp.db`는 없음
  - splitter 객체(`sp, sp_v, splitter_params`)는 ref-직렬화 불가라 `{exp_path}/__splitters.pkl`에 pickle

### DataCache (`_cache.py`)
- `cachetools.LRUCache` 기반, 용량(bytes) 단위 관리
- `get_data(node, typ, idx)`, `put_data(node, typ, idx, data)`
- `clear_nodes(nodes)`: 특정 노드들의 캐시 삭제

### NodeStore (`_store.py`)
- fold 경로 아래 노드별 아티팩트 관리: `{path}/{node_name}/`
  - `obj.pkl` — processor 객체
  - `result.pkl` — fit_transform/fit_predict 출력
  - `info.pkl` — `{status, build_id, role, definition, fit_time, edges, train_shape, ...}`
    - **`role`**: `'stage'`/`'head'` — 아티팩트가 자기 종류를 스스로 설명. `DataFlow.load()`가 이걸로 Trial을 걸러냄
    - **`definition`**: `_definition_of(attrs)` = `{processor, method, adapter, params, edges}` — Trial staleness 비교에 사용
- `status(name)`: `None`(init) / `'built'` / `'finalized'` / `'error'`
- `get_info(name)`: info dict (lazy cache)
- `finalize(name)`: obj/result 삭제, info status → 'finalized'
- `reset_node(name)`: 디렉토리 전체 삭제, cache 무효화

### DataFlow / TrainDataFlow (`_flow.py`)
- **DataFlow** (NodeStore 상속): 디스크에서 stage processor 로드, 소스 데이터를 stage 그래프로 변환
  - `node_objs`: `{name: (obj, result, info)}`, `_node_edges`: `{name: edges}`
  - **`load()`는 stage만 싣는다** (`info['role'] == 'head'`면 skip) — Trial 아티팩트가 같은 fold 디렉토리에 있어서, 안 거르면 Experimenter를 만들 때마다 학습된 모델이 전부 메모리로 올라옴. Trial 모델은 필요할 때 `load_objs(name)`으로 당겨 씀(`Trainer.process`)
  - `get_data(source_data, edges)` → `{key: data}`
- **TrainDataFlow** (DataFlow 상속): stage 빌드 기능 추가
  - `data_source`: DataWrapperProvider (train/valid/**test** 제공 — `test_idx` 보유)
  - `get_train(edges)`, `get_valid(edges)`, **`get_test(edges)`** — flow 하나로 job의 모든 입력을 만들 수 있어야 `TrialJob`이 자족적이 됨
  - `set_objs(name, obj, result, info)`, `get_missing_stages(pipeline)`

### Trainer (`_trainer.py`)
- 생성자: `Trainer(project, name, data, splitter=None, splitter_params=None, aug_data=None, pipeline_name='pipeline', pipeline_version=None)` — 보통 `project.trainer(name, data, ...)`
- 경로 `{project}/trainers/{name}`, 캐시는 `project.cache` 공유
- `set_pipeline_version(version, pipeline_name=None)`: Experimenter와 동일 — 버전 전환 시 `diff_from`으로 stale 제거
- **`set_trials(trials)`**: 학습할 Trial 리스트 + 그것들이 읽는 stage를 자동 선택(`_recompute_selection`). **Trial은 영속화되지 않음** — 로드 후 다시 호출해야 함
- `trials`, `trial_names()`, `trial_attrs()`, `selected_stages`
- `train_folds`: `[TrainFold]` — split당 `TrainDataFlow` 하나
- `train(n_jobs=1, gpu_id_list=None, logger=None)`: stage 먼저(위상 순서), 그 다음 `TrialJob` 실행
- `get_status(node_name)`: `train_data_flows[0]` 조회
- `process(data, v=None)`: generator, split마다 head output을 `v`(DSL 문자열)로 필터 후 concat하여 yield
- `to_inferencer(v=None)`: 학습된 Processor를 추출하여 Inferencer 생성
- `reset_nodes(nodes)`: 하위 종속 노드 포함 초기화
- 저장/로드: `save()`, `Trainer.load(project, name, data)` (= `project.load_trainer(...)`) — `{path}/__trainer.pkl`에 splitter/split_indices + `(pipeline_name, pipeline_version)` 포인터. Trial은 미저장

### Inferencer (`_inferencer.py`)
- 생성자: `(node_attrs, selected_stages, selected_heads, n_splits, node_objs, v=None)`
- **Pipeline 의존성 없음** — `node_attrs`(`{name: attrs}`)만 보유. 실제로 필요한 건 `edges`뿐이라 배포 아티팩트가 순수 데이터가 됨
- `node_objs`: `{name: [processor_split0, processor_split1, ...]}` — Processor 리스트 (Trainer 독립)
- `process(data, agg='mean', nodes=None)`: split 결과 자동 집계
  - `agg`: `'mean'`/`'mode'`/callable/`None`(list 반환). 단일 split이면 집계 없이 반환
  - `nodes`: str/list — 출력할 head 노드 선택 (None=전체). 미등록 노드 지정 시 ValueError
- 저장/로드: `save(path)`, `load(cls, path)` — 단일 `__inferencer.pkl`에 node_objs 포함

### Connector (`_connector.py`)
- `__init__(node_query=None, edges=None, processor=None, role=None)` — 4요소 선택적 매칭
- `processor`: **`"module.ClassName"` 문자열만** (클래스 인스턴스 아님) — resolve 안 하고 그대로 저장
- `match(node_attrs)`: 설정된 요소만 검사, 모두 충족 시 True
  - node_query: str(regex) 또는 list(in), processor: `node_attrs['processor']`(문자열, Pipeline도 항상 문자열로 저장)와 **문자열 그대로 비교**(정규화 없음 — `set_grp`/`set_node`에 준 것과 같은 문자열 형태를 넘겨야 매칭), role: 'stage'/'head' 일치 검사 (None이면 무시)
  - edges: `{key: dsl_string}` — 각 key에 대해 노드의 resolved `edges[key]` 문자열과 **정확히 일치**해야 함 (contain 기반 아님)

### Collector (`collector/` 패키지)
- **Collectors** (`_registry.py`): Collector 인스턴스를 소유하는 레지스트리. `Project.collectors()`로 얻음
  - `Collectors(path=None)` — path 있으면 등록 시 `{path}/{name}`이 기본 저장 위치
  - `set_collector(name, collector, connector, path=None, params=None, exist='skip')` — 부품에서 조립. `collector`는 클래스 또는 `"module.ClassName"`, `connector`는 인스턴스 또는 `{__ref__}`, `params`엔 `resolve_ref_values` 적용
  - `get_collector`/`remove_collector`/`names()`/`in`/`len`/`iter`
  - **`resolve(names)`**: 미등록 이름이면 `KeyError` — 조용히 넘어가면 "아무것도 수집 안 됨"과 구분이 안 되기 때문
  - `match(node_attrs, names=None)`, `save()`, `load(path)` (`__collectors.json`에 name→클래스 ref + path)
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
  - context: `{node_attrs, processor, spec, input, output_train, output_valid}`

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
serial 비교가 아니라 **버전 간 구조 비교**로 판정한다. 판정 지점은 `set_pipeline_version()` **한 곳**이고, `build()`/`train()`은 "디스크에 있는 건 유효하다"를 전제할 수 있다.

**`Pipeline.diff_from(old)`** — DataSource에서 위상 순서로 내려가며:
1. 이름이 old에 없으면 → stale
2. 정의(`_definition_of` = `processor/method/adapter/params/edges`)가 다르면 → stale
3. 정의는 같아도 **읽는 노드 중 하나가 stale이면** → stale (하위 전파가 위상 순서에서 자동으로 나옴)
4. old에는 있는데 지금 없는 이름 → stale (아티팩트 청소용)
5. DataSource schema/targets가 바뀌면 → 전부 stale

- **Trial 처리**: Trial은 Pipeline 밖이라 diff가 이름을 모름. 알 필요도 없음 — 각 아티팩트가 `info['edges']`와 `info['role']`을 기록하므로 stale stage를 읽은 Trial을 **디스크에서 찾아낸다**(`_drop_stale`)
- **Trial 자신의 정의 변경**은 `_make_jobs`에서 `info['definition'] != _definition_of(attrs)`로 **값 직접 비교**. params가 평문 데이터로 강제돼 있어서 dict 비교가 정확함 (해시 불필요)
- **`serial` 자체가 없음(2026-08-01 전체 제거)**: 예전엔 정의 변경마다 새 UUID를 부여해 "뭔가 바뀌었다"는 신호로 썼지만, "결과가 달라지나"는 말 못 했다 — 지금은 `diff_from`이 정의를 직접 비교하므로 그 신호 자체가 불필요해짐. 이게 가능해진 건: **자기가 읽지 않는 stage를 고쳐도 Trial이 유지된다**(예전 serial은 전역이라 이 구분이 안 됐음)는 성질이 부산물로 따라옴

## Lazy resolution: processor / adapter / params (edges DSL과 동일한 지연 원칙)
- `set_grp`/`set_node`는 `processor`/`adapter`/`params`를 **스펙 형태만 검증하고 절대 resolve/instantiate하지 않음** — 실제 값으로의 변환은 전부 사용 시점(`_node_processor.py`)으로 미룸
- **산 객체는 정의 시점에 `TypeError`로 거부됨** (`_validate_processor`/`_validate_adapter`/`_validate_params`, `_pipeline.py`). 에러 메시지가 써야 할 ref 형태를 그대로 안내함
- **processor**: `"module.ClassName"` **문자열만** (클래스 객체 거부). 실제 클래스로 resolve되는 유일한 지점은 `_node_processor.py`(`TransformProcessor`/`PredictProcessor.__init__`의 `resolve_processor()` 호출). `Connector`, `_describer.py`, `_executor.py._needs_gpu`는 전부 이 문자열을 그대로 다룸
  - **클래스를 허용하면 안 되는 이유**: `Connector.match`는 `node_attrs['processor']`를 **문자열 그대로 비교**하므로, 클래스로 정의된 노드는 문자열 ref로 설정한 Connector와 **영영 매칭되지 않음** — 에러 없이 collector가 조용히 아무것도 수집하지 않음. `serialize_value`가 클래스를 `{"__type__":"class"}`로 저장하고 리로드 시 클래스로 되돌리므로 재시작해도 해소되지 않음
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
  - `_executor.py._needs_gpu`(멀티 워커 GPU 디스패치 스케줄링)도 이 함수로 resolve하되, 노드별 결과를 `_gpu_cache`(node name → bool)에 캐싱해 매 dispatch tick 재계산 방지
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
  - `_build_flow_single/multi(outer_folds, pipeline, nodes, ...)` — stage
  - **`_experiment_single/multi(jobs, ...)`** — `TrialJob` 리스트를 그대로 실행. fold 순회/상태 체크/GPU 분류가 executor에서 빠졌고, `gpu_jobs`/`cpu_jobs`는 `job.need_gpu`로 갈림
  - `TrialJob(trial, attrs, cache_key=(outer,inner), flow, need_gpu)` — `flow` 하나로 train/valid/test/ext 입력을 전부 만듦
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
  experimenters.db                  # experimenters (name PK, data_key, title, status,
                                    #                pipeline_name, pipeline_version)
  trials.db                         # trials + experiment_hist

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
    __folds/{outer_idx}/{inner_idx}/{name}/
      obj.pkl                       # processor 객체
      result.pkl                    # fit_transform/fit_predict 출력
      info.pkl                      # {status, build_id, role, definition, edges, ...}
      # stage와 trial이 같은 디렉토리를 쓰지만 NodeStore는 fold당 하나뿐 —
      # 종류 구분은 info['role']이 한다

  trainers/{name}/
    __trainer.pkl                   # splitter, split_indices, (pipeline_name, pipeline_version)
    {split_idx}/{name}/             # obj.pkl / result.pkl / info.pkl

  inferencers/{name}/
    __inferencer.pkl                # node_attrs, selected_stages/heads, n_splits, node_objs, v
```
- Experimenter/Trainer 디렉토리에 **`pipeline.pkl` 사본이 없다** — 포인터만 저장하고 Pipeline은 프로젝트에 한 벌만 존재

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
