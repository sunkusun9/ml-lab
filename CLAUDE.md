# CLAUDE 동작
니가 구사한 코드는 왠만한 건 다 파악 가능해. 주석은 나중에 한꺼번에 만들꺼야, 만들지마
함수나 메소드 가이드도 나중에 할꺼야.

CLAUDE.md에서 불필요하게 토큰을 낭비 하지 않도록, 작업 내역의 개요를 확인해라
작업 관리는 GitHub Issues로 한다. TODO.md 같은 파일은 만들지 마라.
Git 관련 내용(커밋 메시지, PR, 이슈 코멘트)은 영어로 작성한다.
커밋 메시지에 "Co-Authored-By" 넣지 마라. PR에 "Generated with Claude Code" 같은 광고성 메시지 넣지 마라.

코드 검증은 `tests/`에서 적절한 `.py`를 찾아 테스트 케이스를 추가하여 진행한다. `python -c` 같은 임시 실행은 하지 않는다.

**이 문서는 현재 구조만 기술한다.** 변경 이력·날짜·"예전엔 이랬다"는 적지 않는다 — 그건 git log와 GitHub Issues의 몫이다.
설계 근거(왜 이 모양인가)는 현재 코드를 이해하는 데 필요할 때만 현재형으로 남긴다.

## CLI 버전
- git 2.43.0
- gh 2.45.0

## gh CLI 주의사항
- `gh issue view <num>` 는 Projects Classic 지원 deprecated 경고로 exit code 1 반환 → **반드시 `--json` 플래그 사용**
  - 예: `gh issue view 40 --json title,body,comments`
- `--repo` 플래그 없이도 현재 디렉토리의 remote origin에서 자동 추론됨

# mllabs 모듈 요약

## 아키텍처 개요
```
Project(path, cache_maxsize)          경로·캐시 소유. 프로젝트 전역인 것만
  ├─ PipelineBuilder ──build()──► Pipeline    가변 정의 → 불변 노드 그래프
  ├─ TrialStore                     Trial 정의 + 실행 이력 (프로젝트 전역 — 결과 비교가 목적)
  ├─ ProjectStore                   run 이름 목록만 (experimenters / trainers)
  ├─ Experimenter(name)             CV 실험 (exp/{name}/) — Trial을 평가
  │    ├─ ExperimenterStore         meta + splitter + pipeline.pkl (이 run만)
  │    └─ Collectors ──CollectorStore──► Collector 정의(entity 행 + params pkl) → 재조립
  │         └─ CollectHist          수집 이력 (이 run만)
  └─ Trainer(name)                  전체 데이터 학습 (trainers/{name}/) — Predictor를 학습
       ├─ TrainerStore              meta + splits + pipeline.pkl (이 Trainer만)
       ├─ PredictorStore            Predictor 정의 (이 Trainer만 — 비교할 일이 없어서)
       └─ to_inferencer() ──► Inferencer
```
**경계**: run 하나에 대한 것(splitter, 채택한 Pipeline, 아티팩트/이력, Collector, 학습할 Predictor)은
전부 그 run 디렉토리 안. Project는 "이 프로젝트에 뭐가 있나"만 답한다 → **어떤 run이든 Project 없이
열 수 있다** (`Experimenter.load_experimenter(path, data)` / `Trainer.load_trainer(path, data)`).

- **Project** (`_project.py`): 디렉토리 레이아웃 + `TrialStore`/`ProjectStore`/`cache`. 프로젝트 전역인 것만 소유하고, Pipeline 버전조차 색인하지 않는다(각 pipeline이 자기 db에서 관리)
- **PipelineBuilder / Pipeline** (`_pipeline.py`): 가변 빌더 + `build()`가 만드는 불변 **노드 전용** 그래프
- **Trial / make_trials** (`_trial.py`): 평가할 구성 하나. Pipeline 밖에 있음
- **TrialStore** (`_trial_store.py`): `trials`(정의) + `experiment_hist`(fold별 실행 이력). 저작은 `Project.set_trial`, 실행은 `Experimenter.exp`가 **이름으로** 꺼내 씀
- **Predictor** (`_predictor.py`): Trainer가 학습하는 끝지점 출력 노드. Trial과 같은 실행 정의 + 출처(`src_trial`/`src_experimenter`)
- **PredictorStore** (`_predictor_store.py`): `predictors`(정의) 하나뿐 — 이력은 Trainer의 두 번째 `NodeStore`가 가짐
- **ProjectStore** (`_project_store.py`): `experimenters`/`trainers` **이름 목록만**
- **ExperimenterStore / TrainerStore** (`_experimenter_store.py`, `_trainer_store.py`): run 하나 전용 상태(meta + splitter BLOB + `pipeline.pkl`)
- **Experimenter** (`_experimenter.py`): CV 실험 실행/관리. 생성자=신규, 복원=`load_experimenter()`
- **Trainer** (`_trainer.py`): 학습 실행/관리 (split 기반). 생성자=신규, 복원=`load_trainer()`, 학습 대상은 `train(predictors)`로 직접 전달
- **Inferencer** (`_inferencer.py`): 학습된 processor를 새 데이터에 적용
- **NodeStore** (`_store.py`): run 하나당 하나 — 노드 아티팩트(obj.pkl/result.pkl) + 실행 이력(`node_hist`) 둘 다 소유
- **DataFlow / TrainDataFlow** (`_flow.py`): fold별 데이터 흐름 및 노드 빌드 (NodeStore를 컴포지션으로 보유, outer_idx/inner_idx도 보유)
- **_executor.py**: `_execute_single`(단일 프로세스) + `_execute_multi`(멀티 워커) — 노드/Trial/Predictor 공용. `store`(그 job 종류의 기록을 소유한 store)와 `chained`(job들이 서로 먹이는가)로 갈림

## Node/Trial 상태 모델

### 3-State
`init → built` / `init → error → (reset) → init`

| 상태 | Disk | 설명 |
|------|------|------|
| **init** | - | 정의만 된 상태 |
| **built** | O | 빌드 완료, 결과 추출 가능 |
| **error** | info only | 실행 중 에러 발생, 내역 보존 |

Disk 칸은 **노드(와 Trainer의 Predictor)** 얘기다 — Trial은 아래 참조

- 중간 상태(finalize)는 없다. 아티팩트를 없애는 방법은 `reset_nodes()`(완전 삭제, `init`으로 복귀) 하나뿐
- **Experimenter엔 상태 게이트가 없다** — `open`/`close`/`status` 개념 자체가 없고 `build()`/`exp()`는 언제나 호출 가능
- **Trial은 아티팩트를 안 남긴다** — 위 표에서 Trial에 해당하는 Disk 칸은 항상 비어 있다. 평가받는 후보라서 남길 가치가 있는 건 모델이 아니라 결과이고, 그 결과는 `experiment_hist`(+ Collector가 모은 데이터)에 있다. 그래서 Trial엔 `built`/`init` 구분이 디스크에 없고, `reset_nodes()`가 지울 것도 없으며, 재실행 수단은 `TrialStore.remove_hist(...)` 하나뿐 ("Staleness" 섹션)

## 핵심 클래스

### Node 역할
- **DataSource** (`_DataSourceNode`, key=`None`): 원본 데이터 스키마 및 target 정의
- **노드**: 전처리/변환 (TransformProcessor). **Pipeline에 담기는 유일한 종류** — 그래서 별도 명칭 없이 그냥 "노드"
- **Trial**: 모델링/예측 (PredictProcessor). Pipeline 밖, Experimenter 쪽 — `_trial.py`
- **Predictor**: 같은 자리(끝지점 출력)의 Trainer 쪽 대응물 — `_predictor.py`. Trial과 **일부러 별도 클래스**(아래 "Predictor" 섹션)

### PipelineBuilder / Pipeline 분리 (`_pipeline.py`)
```
PipelineBuilder  — 가변. grps 계층, SQLite(pipeline.db), set_grp/set_node
  └─ .build() ──► Pipeline  — 불변 스냅샷. grp 상속 해소 완료, 순수 데이터
```
- **Experimenter/Trainer/Inferencer는 `Pipeline`만 보유** — builder를 넘기면 `TypeError`(`_run_common.require_built_pipeline`). builder를 나중에 수정해도 진행 중인 실행에 새어 들어가지 않음
- **Pipeline은 노드 전용** — `role` 개념이 코드 어디에도 없다. 노드와 Trial을 구분하는 필드 자체가 없고, 구분이 필요한 자리도 없음(Collector는 Trial job에만 붙는다)
- grp는 build를 넘어가지 않음 — 원래 그룹명은 표시용 `label`로만 남음 (`_BuiltNode.label`에만 있고 `ProcessorSpec`엔 `grp`/`label` 둘 다 없음)
- 노드에 `tag` 없음 (Trial 쪽에만 있음)

#### PipelineBuilder
- `VAR_TYPES = frozenset({'numerical', 'ordinal', 'nominal', 'text', 'binary', 'datetime'})`
- `_params_equal(a, b)`: `a == b` 한 줄 — params가 순수 데이터/ref spec만 담도록 강제되므로 `__dict__` 재귀 비교 같은 우회가 필요 없음
- `nodes`: `{name: _PipelineNode}` (`None` → `_DataSourceNode`), `grps`: `{name: _PipelineGroup}` (`'__datasource__'` 항상 존재)
- `datasource`: `nodes[None]` 반환 property
- `set_datasource(schema, targets=None)`: DataSource 스키마/target 설정
- `set_grp(exist='diff'|'skip'|'error'|'replace')`, `set_node(exist=...)`, `rename_grp`, `remove_grp`, `remove_node`
  - `role`/`tag` 파라미터 없음
  - **`processor`/`adapter`/`params` 스펙 검증** (`_validate_processor`/`_validate_adapter`/`_validate_params`) — 산 객체를 넘기면 `TypeError`. "Lazy resolution" 섹션 참조
- `build()` → `Pipeline`
- `get_node_names(query)`, `get_node_spec(name)`, `_find_descendants(name)`
- `sync()`: DB가 source of truth. 그룹/노드 필드를 직접 값 비교(`diff()`)해 갱신하고, **그룹이 바뀌면 그 그룹(+자식 그룹) 소속 노드들의 attrs 캐시도 함께 무효화**해 `changes['nodes']['updated']`에 포함시킴(노드 자신의 행은 안 바뀌었어도 상속받는 값이 바뀌었으므로)
- **`serial` 없음** — 정의 변경을 표시하는 전역 신호를 두지 않는다. staleness는 `Pipeline.diff_from`의 값 비교, 버전은 `PipelineStore`의 단순 `max+1` 카운터가 담당(해시/dedup 없음)
- `copy()`, `copy_nodes(node_names)` — 선택적 복사 (builder→builder)
- `compare_nodes(nodes)` → `{processor_name: DataFrame}` (params 차이 + edges['X'] 노드별 변수 차이)
- `desc_pipeline(max_depth, direction)`, `desc_node(node_name, direction, show_params)`: Mermaid 다이어그램 — grp 계층이 필요하므로 **builder 전용**

#### Pipeline (빌드 결과)
- `nodes`: `{name: _BuiltNode}` — `None` 키는 `_BuiltDataSource` (builder와 동일한 관례)
- `_BuiltNode` 속성(`__slots__`): `name`, `label`, `processor`, `edges`, `method`, `adapter`, `params`, `desc`, `output_edges`
- `pipeline_id`(builder 신원) / `build_id`(빌드 호출마다 새 UUID) / `version`(`int | None`) — **`Project.build_pipeline()`이 저장할 때만 세팅**. `builder.build()`를 직접 부르면 `None`(미저장 in-memory 빌드)
- `get_node(name)`, `get_node_spec(name)`(ProcessorSpec — 노드 전용, DataSource는 `pipeline.datasource`로), `get_node_names(query=None)`
- `topo_order()`: DataSource에서 내려오는 깊이순 노드명 (DataSource 제외) — 빌드 시 1회 계산해 캐시
- `descendants(name)`, `check_data_compatibility(data)`
- `diff_from(old)` → `set[str]`: "Staleness" 섹션 참조
- `subset(node_names)`: 지정 노드 + 조상만 담은 새 Pipeline
- **불변성의 한계**: `params`/`edges`는 shallow copy — 중첩 값은 builder와 공유. "수정하지 않는다"는 관례로 지킴

#### 내부 노드 클래스
- **`_DataSourceNode`** (`_PipelineNode` 서브클래스):
  - `schema`: `{col: var_type}` — var_type은 VAR_TYPES 중 하나
  - `targets`: `list[str]` — 타겟 컬럼 목록 (타입과 별도)
  - `get_attrs(grps=None)`: `name`/`grp`/`schema`/`targets` **dict** 반환 — DataSource는 실행 대상이 아니라 만들 Processor가 없어서 `get_spec`을 오버라이드하지 않는다(같은 이름의 다형 메소드가 서로 다른 모양을 반환하는 걸 피하려고 이름을 분리)

- **`_PipelineGroup`**: 노드 그룹 — builder 내부 전용
  - 속성: `name`, `processor`, `edges`, `method`, `parent`, `adapter`, `params`, `desc`
  - `children`: 자식 그룹명 리스트, `nodes`: 소속 노드명 리스트
  - `get_attrs(grps)`: 상위 그룹 속성 병합하여 **dict** 반환 (`desc`는 상속 안 됨) — 그룹은 실행 단위가 아니라 상속 해소용이라 `ProcessorSpec`이 아님. 캐시는 `self.attrs`/`update_attrs()`
  - `diff(processor, edges, method, parent, adapter, params)`: 달라진 필드명 리스트 반환 (`desc` 제외 → desc-only 변경은 rebuild 미유발)

- **`_PipelineNode`**: 개별 노드 — builder 내부 전용
  - 속성: `name`, `grp`, `processor`, `edges`, `method`, `adapter`, `params`, `desc`
  - `output_edges`: 이 노드를 입력으로 사용하는 노드명 리스트
  - `get_spec(grps)`: 그룹 속성과 노드 속성을 병합해 `ProcessorSpec` 반환(캐시 `self.spec`, 무효화 `update_spec()`)
  - `diff(grp, processor, edges, method, adapter, params)`: 달라진 필드명 리스트 반환 (`desc` 제외)
  - `set_grp`/`set_node`: `desc` 파라미터 수락; exist='diff' skip 경로에서도 `desc`는 업데이트됨

- **ColSelector** (`_pipeline.py`): processor params(예: `cat_features`, `cat_cols`)에 쓰는 지연(lazy) 컬럼 선택자
  - `__init__(dsl_string='*')` — DSL 문자열 하나만 보유(정의 시점엔 데이터 불필요, `edges[key]`와 동일한 원칙)
  - **params에는 인스턴스가 아니라 ref-dict로 지정**: `{"__ref__": "mllabs.ColSelector", "__params__": {"dsl_string": "*@categorical"}}` (인스턴스는 `set_grp`/`set_node`가 `TypeError`로 거부)
  - `_node_processor`가 Processor 생성 시 `resolve_ref_values()`로 인스턴스화하고, fit 시점에 `_resolve_col_selectors`가 `eval_expr(parse(v.dsl_string), data)`로 컬럼 확정

### ProcessorSpec (`_pipeline.py`)
**노드/Trial/Predictor가 공통으로 resolve되는 단 하나의 실행 단위 표현.** 필드는 정확히 6개:
`name`, `processor`, `edges`, `method`, `adapter`, `params` (`__slots__`, immutable 취급, 값 기반 `__eq__`)

- 이 중 **5개(`name`/`processor`/`method`/`adapter`/`params`)가 Processor 생성자 인자 그대로** — `_node_processor.py`의 `TransformProcessor`/`PredictProcessor`가 받는 것과 1:1
- **`edges`는 Processor한테 안 넘어감** — flow가 "무엇을 먹일지" 정하는 입력 배선이고, 실행 시점에 실제 데이터에 대해서만 lazily 컬럼으로 확정됨. 이름이 `ProcessorAttr`가 아니라 `ProcessorSpec`인 이유("아직 resolve 안 된 선언")
- **표시 전용 필드는 일부러 뺌** — 노드의 `label`(원래 grp명), Trial의 `desc`/`tag`. 전부 원본 객체에서 직접 꺼낼 수 있음
- 만드는 쪽: `_BuiltNode.get_spec()`, `_PipelineNode.get_spec(grps)`, `Trial.get_spec()`, `Predictor.get_spec()`
- 쓰는 쪽: `Job.spec` → `_process()`(Processor 생성) / `_definition_of()`(staleness·info) / flow의 입력 준비 / `Connector.match(spec)` / `_describer` / `Inferencer.node_specs`
- **DataSource는 여기 안 들어감** — `_DataSourceNode.get_attrs()`/`_BuiltDataSource.get_attrs()`가 `name`/`grp`/`schema`/`targets` **dict**를 반환. 접근은 `pipeline.datasource` / `builder.datasource`
- `_PipelineGroup.get_attrs(grps)`도 dict — 그룹은 실행 단위가 아니라 상속 해소 중간 단계

### Trial (`_trial.py`)
평가할 구성 하나. **Experiment 클래스는 없음** — Trial 리스트를 직접 넘긴다.

- **`Trial`**: `name`, `processor`, `method`, `adapter`, `params`, `edges`, `desc`, `tag`
  - `desc`: 순수 표시용 설명 문자열 — `PipelineBuilder`의 `desc`와 같은 역할(매칭/diff/저장 식별에 전혀 관여 안 함). 안 주면 `None`
  - `get_spec()`: 노드의 `Pipeline.get_node_spec()`과 **똑같은 `ProcessorSpec`** 반환
  - **이름이 식별자**. `TrialStore`(`trials` 테이블 PK)의 키이자 `experiment_hist`가 매다는 것이고, `exp()`가 참조하는 것도 이 이름 — Trial 객체는 프로젝트에 등록될 때만 등장하고 실행 경로엔 이름만 다닌다
  - 정의를 값으로 비교하는 별도 유틸(content_key 류)은 없다 — `TrialStore.has()`가 필드별로 직접 비교. 이 비교가 `Project.set_trial`의 문지기(변경 여부 + 성공 이력 있는 이름 동결)를 떠받침
  - `node_names()`: edges가 참조하는 노드 이름 집합

- **`make_trials(name, processor, edges, method, adapter, params, param_grid, tags)`** → `list[Trial]`
  - `params`(전 trial 공통) + `param_grid`(`{param: [values]}`) 카테시안 곱, grid 키 정렬 기준 결정적 순서
  - 이름: 단일이면 `{name}`, 복수면 `{name}_{idx}` (0 패딩)
  - `_validate_processor`/`_validate_adapter`/`_validate_params`로 spec 검증 (Pipeline과 동일 규칙)

### Project (`_project.py`)
디렉토리 레이아웃 소유 + **프로젝트 전역인 것만** 소유. **모든 컴포넌트가 단독 동작 가능** —
Project는 순수 "조각을 짜맞춰주는 팩토리"일 뿐 필수 의존성이 아니다. `experimenter()`/`trainer()`가 하는 일:
1. 경로(`exp_path`/`trainer_path`)와 `cache` 제공
2. `ProjectStore`에 이름 등록
3. `pipeline_version`을 줬으면 `load_pipeline()`으로 `Pipeline` 객체로 바꿔 채택시킴

`load_experimenter()`/`load_trainer()`는 **Pipeline을 resolve하지 않고** 각 클래스의 복원 진입점에
위임한다 — run이 자기 디렉토리에서 다 읽어온다. 그래서 어떤 run이든 Project 없이 열 수 있고,
ProjectStore는 이름 목록이라 동기화가 어긋날 두 번째 진실 원본이 되지 않는다.

- `Project(path, cache_maxsize=4GB)` — `DataCache`를 소유하고 모든 Experimenter/Trainer가 공유
- 경로: `pipeline_path(name)`, `exp_path(name)`, `trainer_path(name)`, `inferencer_path(name)` — **collector 경로 없음**(레지스트리는 run 소유)
- 팩토리: `pipeline_builder(name)`, `experimenter(name, data, pipeline_name=, pipeline_version=, **kw)`, `load_experimenter(name, data, data_key=, aug_data=)`, `trainer(name, data, pipeline_name=, pipeline_version=, **kw)`, `load_trainer(name, data, aug_data=)`
- **Pipeline 버전**: `build_pipeline(builder)` → `builder.build()` 후 결과를 다음 버전(1부터, `max+1`)으로 저장하고 `pipeline.version`에 세팅해 반환. **content dedup 없음** — 내용이 같아도 호출할 때마다 새 버전(`builder`에 path가 없으면 `ValueError`)
  - 카운터/버전 파일은 **각 pipeline 자신의 db**(`pipelines/{name}/{name}.db`)가 소유 — `build_pipeline`은 `builder._store.save_version()`에 위임할 뿐, 프로젝트 전역 색인이 없음
  - `load_pipeline(name, version=None)`, `list_pipeline_versions(name)` — 둘 다 `PipelineStore(pipeline_path(name), name)`를 통해 조회
  - 저장은 pkl (`v{n}.pkl`) — 형식은 `PipelineStore.save_version`/`load_version` 뒤에 숨어 있어 교체 가능
- `trials`: `TrialStore`, `store`: `ProjectStore`, `list_experimenters()`, `list_trainers()`
- **`set_trial(trial)` / `set_trials(trials)`**: Trial 저작 진입점. **실행과 분리된 이유** — Trial은 프로젝트 소유인데 등록이 `Experimenter.exp()`의 부수효과였다. 그래서 실행하지 않고는 프로젝트에 넣을 수 없었고, 한 run이 다른 run이 이미 쓴 이름을 조용히 덮을 수 있었다
  - 반환은 **추가·변경된 이름**(`set_trial`은 str 또는 `None`, `set_trials`는 list) — 그대로 다음 `exp()`의 작업 목록이 된다. 저장된 정의와 같으면 변경이 아님(`has()`)
  - **성공 이력이 있는 이름은 얼린다**: `experiment_hist`에 `'built'` fold가 있는데 정의가 다르면 `ValueError`. 이력은 이름으로 키잉되고 Trial은 아티팩트를 안 남기므로, 재정의하면 옛 결과가 그걸 만든 적 없는 정의를 설명하게 된다 — 한 Trial의 기록처럼 보이는 두 개가 됨. 바꾸려면 새 이름이거나 `remove_trial`(결과까지 포기). `'error'`뿐이면 지킬 결과가 없으니 허용
  - `set_trials`는 **전부 검사한 뒤에 쓴다** — 하나가 얼려 있으면 아무것도 안 바뀐다(절반만 등록되면 반환한 작업 목록이 거짓이 됨)
  - 문지기 없는 저수준 경로는 `trials.register()` — 테스트가 재정의 동작을 직접 확인할 때 씀
- **`show_error_trials(experimenter=None, traceback=False)`**: Trial 실행 에러를 fold당 한 줄로. `Experimenter.show_error_nodes`의 Trial판이고 **여기 있는 이유는 이력의 주인이 여기라서**(노드 이력은 run, Trial 이력은 프로젝트). 실패가 없으면 `None`
- **`pending_trials(experimenter=None)`**: 등록된 Trial 중 **에러났거나 이력이 아예 없는** 이름 목록. 둘 다 "아직 돌려야 할 것"이라 한 목록으로 낸다 — 손으로 쓰면 후자만 잡아서 실패한 게 조용히 빠진다(#130이 노트북에서 지목한 실제 버그)
  - **일부러 거칠다**: fold 일부만 돌다 끊긴 건 안 잡는다 — 그걸 판정하려면 fold 그리드와 대조해야 하는데 이 store는 그걸 모른다. 반환된 이름을 그냥 돌려도 안전하다(`exp()`가 끝난 fold를 건너뜀)
- **`remove_trial(name, experimenters=None)`**: Trial 하나를 프로젝트에서 완전히 지운다 — 정의(`TrialStore.trials`) + 이력(`experiment_hist`, **모든 experimenter**) + 각 run이 수집한 데이터와 그 `CollectHist`. Trial은 아티팩트를 안 남기는 대신 흔적이 서로 모르는 store들에 흩어져 있고, **그걸 다 보는 건 Project뿐**이라 여기 있다(단일 store 위의 편의 래퍼가 아니라, 주인 없는 교차 연산에 주인을 준 것)
  - 프로젝트 전역 절반(정의 + 전 experimenter 이력)은 각각 SQL 한 문장이고, run별 절반은 `list_experimenters()` 순회 → `Experimenter.remove_trial_result(name)` 위임
  - **run을 열지 않는다** — `Collectors({exp_path}/collectors)`로 레지스트리만 경로에서 연다(db 두 개뿐, 데이터셋 불필요). Experimenter를 만들면 `DataFlow.__init__`이 그 fold의 아티팩트를 전부 적재하므로 trial 하나 지우자고 치를 비용이 아님
  - `experimenters=`: **이미 열어둔 run이 있으면 그걸 넘길 것** — 경로로 새로 연 레지스트리는 내가 든 인스턴스가 아니고, 일부 Collector는 메모리 캐시에서 답한다(`ModelAttrCollector`/`SHAPCollector`의 `_cache`)
  - 특정 Experimenter만 다시 돌리고 싶은 거라면 이게 아니라 `e.remove_trial_result(name, trial_store)`

### ArtifactStore (`_store.py`, 공통 인터페이스)
`NodeStore`/`TrialStore`가 공유하는 메소드 모양의 base class. 두 그룹:
- **아티팩트** (`write_objs`/`write_obj`/`write_result`/`get_objs`/`get_obj`/`get_result`/`list_nodes`/`status`/`reset_node`) — **`NodeStore`만 전부 구현**. `TrialStore`는 상속만 하고 하나도 오버라이드하지 않음 — Trial은 아무 데도 아티팩트를 안 남기므로 서빙할 obj/result 자체가 없음. 상속해두는 이유는 base가 `NotImplementedError`를 던져서, `TrialStore`에서 호출하면 `AttributeError` 대신 의도가 분명한 에러가 나기 때문
- **`stores_artifacts`**(클래스 속성, base `False` / `NodeStore` `True`): 위 두 그룹 중 어느 쪽을 실제로 구현하는지를 코드가 읽을 수 있는 형태로 만든 것. 덕분에 호출부는 executor에 "그 job 종류의 기록을 소유한 store"를 그냥 넘기고, 저장할 게 있는지는 executor가 store에 물어본다
- **히스토리** (`record`/`get_hist`/`get_status`/`get_info`/`remove_hist`) — 둘 다 각자 자기 테이블에 대해 구현. `TrialStore`가 (이미 한 run에 스코프된 `NodeStore`엔 없는) experimenter 이름을 키로 하나 더 쓰기 때문에 override 시그니처가 서로 달라, base에선 `*args, **kwargs`로만 선언

### TrialStore (`_trial_store.py`)
```sql
trials(name PK, desc, processor, method, adapter, params, edges, tag)
experiment_hist(trial_name, experimenter, outer_idx, inner_idx,  -- PK
                pipeline_version, status)
```
- **인조식별자도 content hash도 없다.** 두 테이블 다 **이름이 PK**(`trials`는 trial 이름, 이력은 trial 이름 + experimenter 이름). `pipeline_version`은 해시가 아니라 **정수**로, 그 실행의 `Experimenter.pipeline_version`을 그대로 기록
- 이름으로 키잉하는 이유: Experimenter가 이미 이름으로 키잉돼 있어(`{project}/exp/{name}`) 조인 없이 읽히고, **이름을 재정의하면 행을 덮어쓴다**가 두 테이블 모두 일관됨(`register`는 `INSERT OR REPLACE`)
- 정의 일치 여부는 값 비교 하나로 충분해서(`has()`) 해시 컬럼을 두지 않는다. `experiment_hist`는 실행 로그일 뿐 정의의 출처가 아니므로, 이름이 재정의되면 예전 정의를 복원하는 기능은 없다
- **재실행 여부는 `experiment_hist`가 판정한다** — `Experimenter._make_jobs`는 `experiment_hist`의 fold별 `status`만 본다: `'built'`면 스킵, `'error'`거나 기록이 없으면 job 생성. **디스크에 이와 어긋날 것이 애초에 없어서**(Trial은 아티팩트를 안 남김) 이게 유일하게 가능한 판정이고, 다시 돌리려면 `e.remove_trial_result(name)` 하나면 된다. 재정의해도 이미 `'built'`인 fold는 자동 재실행되지 않으며, **애초에 그런 재정의가 `Project.set_trial`에서 막힌다**
- `register(trial)`/`register_all(trials)`: 이름 기준 upsert — **문지기가 없는 저수준 API**. 성공 이력 검사를 거치는 저작 진입점은 `Project.set_trial`/`set_trials`
- `has(trial)`: 그 이름에 저장된 게 **지금** 이 정의와 같은지 필드별 비교. 저장된 게 아예 없으면 `False`(호출부가 부재를 따로 안 봐도 되게). `desc`/`tag`는 비교 대상이 아님 — 표시·선택용이라 실행이 달라지지 않음
- `get_by_name(name)`/`list_trials()`: **`Trial` 객체**를 반환(`PredictorStore`와 같음). `exp()`가 이름으로 받아 여기서 정의를 꺼내 실행하므로, 넣은 것과 같은 것이 나와야 함
- `remove(name)`: **정의만** 삭제 — `experiment_hist`는 그대로 두어 "정의가 사라진 뒤에도 무엇이 돌았는지는 읽힌다". 프로젝트에서 통째로 걷어내는 건 여러 store에 걸친 일이라 `Project.remove_trial(name)`
- `record(trial_name, experimenter, outer_idx, inner_idx, pipeline_version, status)`, `get_hist(trial_name=, experimenter=, pipeline_version=, status=)`, `get_status(...)`, `remove_hist(...)`

### Experimenter (`_experimenter.py`)
- **Project 의존성 없음. 주입받는 건 `cache` 하나** — 생성자: `Experimenter(path, name, data, data_names=, sp=, sp_v=, splitter_params=, title=, data_key=, aug_data=, cache=)`. store는 `ExperimenterStore(self.path)`로 **자기가 만들고**, Pipeline은 `set_pipeline()`으로만 채택
  - `data`는 native/`DataWrapper` 아무거나 — `self.data = wrap(data)`, splitter에는 `unwrap(data)`를 넘김
  - **생성자 = 신규 생성**. split을 다시 계산하고 상태를 새로 씀 → 기존 디렉토리에 대고 부르면 재개가 아니라 처음부터 다시 시작
  - 복원은 **`Experimenter.load_experimenter(path, data, data_key=None, aug_data=None, cache=None)`** staticmethod. `{path}/__exp.db`가 없으면 `KeyError` — store를 만들기 **전에** 검사한다(안 그러면 없는 run의 디렉토리와 빈 db를 만들어놓고 실패함)
  - `Project.experimenter(...)`가 하는 일은 경로 + `cache` + ProjectStore 이름 등록 + (버전을 줬으면) `set_pipeline` 뿐
- **이름이 식별자**: 경로는 `{project}/exp/{name}`, `TrialStore` 이력의 키도 이 이름. UUID 없음
- **Pipeline은 객체로 지정** — `set_pipeline(pipeline, pipeline_name=None)`이 이미 로드된 `Pipeline`을 받아 채택. 이 클래스는 이름/버전으로 파이프라인을 **로드할 방법 자체가 없다**(project 참조가 없음) — 버전 번호로 지정하려면 `Project.experimenter(..., pipeline_version=)`. `self.pipeline_version`은 별도로 안 들고 `pipeline.version`에서 읽음(단일 출처)
  - 채택한 Pipeline은 **실험 디렉토리의 `pipeline.pkl`로 저장되고 `load_experimenter()`가 그걸 다시 읽는다**(`_experimenter.py:284`) — Project 없이도 마지막에 채택한 Pipeline이 복원됨. **생성자는 안 읽는다** — `self.pipeline = None`으로 두고 `_save()`가 meta의 `pipeline_name`/`pipeline_version`까지 기본값으로 덮어쓰므로, 기존 디렉토리에 대고 생성자를 부르면 provenance가 사라진다(GitHub #128)
  - 버전 전환 시 `pipeline.diff_from(self.pipeline)`으로 stale 판정 → `reset_nodes()`로 해당 노드 아티팩트만 제거. Trial은 건드리지 않음("Staleness" 섹션)
- `cache`(`DataCache`, optional) — `None`이면 캐시 없이 동작. store는 optional이 아니라(자기가 만듦) standalone Experimenter도 meta/splitter/Pipeline이 전부 저장됨
- **`collectors`**: `Collectors({path}/collectors)` — 생성자에서 만들고(생성자가 곧 복원) `node_store`와 같은 자리. Collector가 쓰는 건 전부 노드 이름만으로 키잉되므로 **경로가 두 run을 가르는 유일한 수단**이다 — 프로젝트 전역 레지스트리였다면 이름이 겹치는 Trial마다 서로의 결과를 덮어썼다(무증상, 그리고 하필 비교가 필요한 바로 그 경우에). 대가는 교차 run 비교가 store N개에 대한 읽기가 되는 것(#130)
- **`trial_store`**(`TrialStore`, optional): 실행할 Trial을 읽고 결과를 기록하는 곳. `cache`처럼 **생성자에서 한 번** 주입 — 프로젝트에 하나뿐이고 run 수명 동안 바뀌지 않는다. **영속화 안 함**: 프로젝트 레벨 객체라 자기 디렉토리만 아는 run이 찾아낼 방법이 없고, 상위 레이아웃을 추측하는 건 이 구조가 피해온 암묵 결합. 없으면 `_require_trial_store()`가 두 공급 경로를 안내하며 `RuntimeError`
- **`remove_trial_result(name)`**: 이 run이 가진 Trial *결과*를 지운다 — `self.collectors.remove_results(name)`(수집 데이터 + `CollectHist` 행) + 이 run의 `experiment_hist` 행 → **다음 `exp()`가 그 Trial을 다시 돌린다**(fold 스킵의 유일한 근거가 이력이므로). 다른 run의 기록도, 정의(프로젝트 소유)도 안 건드림. `trial_store`가 없으면 이력 절반은 건너뜀
- **pipeline 필요** (`_require_pipeline()`로 미설정 시 에러):
  - `build(nodes=None, rebuild=False, n_jobs=1, gpu_id_list=None, logger=None)` — 노드 빌드
  - **`exp(trials, collectors=None, n_jobs=1, gpu_id_list=None, logger=None)`**
    - `trials`: **Trial 이름 리스트**. 이름 하나가 이 run의 fold 전 조합을 뜻하며, fold 지정은 호출부의 몫이 아니다
    - **이름이지 인스턴스가 아니다** — 정의는 주입된 `self.trial_store`에서 꺼낸다. Trial은 프로젝트 소유인데 예전엔 등록이 `exp()`의 부수효과라, 실행하지 않고는 프로젝트에 넣을 수 없었고 한 run이 다른 run이 쓰던 이름을 조용히 재정의할 수 있었다. 저작은 `Project.set_trial`, 여기는 실행만
    - 미등록 이름은 `KeyError`, 이름 자리에 `Trial`을 넘기면 `TypeError`(안 잡으면 sqlite 바인딩 에러로 새어나가 원인이 안 보임)
    - `collectors`: **이 run의 `self.collectors`에 등록된 이름 리스트**. `None`=전부, `[]`=수집 안 함. 인스턴스를 넘기면 `TypeError`(`_resolve_collectors`) — 레지스트리가 모르는 Collector는 이 run에 쓸 자리가 없어서 조용히 run 밖에 결과를 떨군다. 미등록 이름은 `Collectors.resolve`가 `KeyError`
    - 수집 이력은 항상 `self.collectors.hist` — 한 호출의 선택이 아니라 run에 속하므로 인자가 없다
    - **`_make_jobs(trials)`가 fold 조합을 만든다** — 이름마다 `(outer_idx, inner_idx)` 전 조합을 돌며 `'built'`인 것만 빼고 `Job(name, spec, outer_idx, inner_idx, flow, need_gpu)` 생성. skip 판정은 `trial_store.get_status(name, self.name)`의 fold별 status로만 하므로, **같은 이름을 다시 넘기면 중단된 실행이 이어진다**(반복이 아니라). 정의 조회·spec·GPU 판정은 fold당이 아니라 이름당 1회
    - `TrialHistTracker`가 fold별 done/error를 이력에 기록(등록은 안 함)
  - `n_jobs`는 실제 작업 수로 상한 처리 (`min(n_jobs, len(jobs))`) — 유휴 워커/progress bar 방지
  - `get_node_info()`: 노드 요약 Markdown
- **pipeline 불필요** (디스크 상태만으로 동작): `get_status(node_name)`, `reset_nodes(nodes)`, `show_error_nodes(nodes=None, traceback=False)`, `get_objs(node_name, outer_idx=0, inner_idx=0)`
  - **넷 다 Pipeline 노드 전용.** Trial은 디스크에 아무것도 안 남기므로 `get_status`는 몇 번을 돌렸든 `None`, `reset_nodes`는 지울 게 없고, `get_objs`는 `FileNotFoundError`. Trial 쪽 대응물은 `trial_store.get_status(name, exp.name)` / `Project.show_error_trials(experimenter=)` / `remove_trial_result(name)` / (모델 대신) Collector가 모은 결과
  - **에러 보고가 갈리는 기준은 이력의 주인이다** — 노드 이력은 run의 `node_hist`, Trial 이력은 프로젝트의 `experiment_hist`. 예전엔 `show_error_nodes` 하나가 둘을 합쳐 내놨는데, 출력 라벨이 `[이름] fold o_i` 하나뿐이라 노드인지 Trial인지 구분이 안 됐다. 포맷은 `_run_common.format_errors`가 공유(둘 다 실패를 같은 모양으로 설명하므로)
- **OS log capture** (`open_os_log`/`close_os_log`/`os_log`):
  - `open_os_log(log_path=None)`: 이 프로세스의 OS-level stdout/stderr(fd 1/2)를 `{path}/__worker_logs/master.log`(기본값)로 dup2 리다이렉트 — `self._os_log_state`에 원본 fd/`sys.stdout`·`stderr` 백업 보관. 이미 open이면 에러
  - `close_os_log()`: 리다이렉트 원복. open 안 된 상태에서 호출하면 no-op
  - `os_log(log_path=None)`: 위 둘을 감싼 컨텍스트 매니저 — `with e.os_log(): e.build(n_jobs=1); e.exp(n_jobs=4)`
  - open~close 구간 동안 `n_jobs=1`인 `build`/`exp`는 같은 프로세스라 마스터 리다이렉트가 그대로 캡처하고, `n_jobs>1`이면 그 구간에 한해 `log_dir`이 전달돼 워커 stdout/stderr도 `{path}/__worker_logs/worker_{i}.log`로 리다이렉트됨
  - `sys.stdout`/`stderr`는 원본 fd의 dup으로 rebind되므로 `DefaultLogger`의 진행률 등 Python 레벨 출력은 콘솔에 그대로 보인다 — dup2로 fd 1/2만 돌리기 때문에 native(C-level) 직접 write만 잡힘
  - `get_worker_logs(worker=None)`: 캡처된 네이티브 출력 — `{worker_idx: text, 'master': text}`. 매 실행마다 덮어씀
- `get_train_data(edges, o_idx=0, i_idx=0)` / `get_valid_data(...)` / `get_test_data(...)`: 출력 추출 헬퍼
- `aug_data`: 외부 데이터를 DataSource 수준에서 inner train split에 append — 미퍼시스트
- 저장/로드: meta/splitter는 **그 run의 `{exp_path}/__exp.db`**(`_experimenter_store.py`) — `experimenter(name PK, data_key, title, pipeline_name, pipeline_version, splitters BLOB)`. 프로젝트 전역 experimenter db는 없다
  - splitter 객체(`sp, sp_v, splitter_params`)는 ref-직렬화 불가라 컬럼이 아니라 **BLOB**(pickle)

### DataCache (`_cache.py`)
- `cachetools.LRUCache` 기반, 용량(bytes) 단위 관리. `Project`가 소유(`project.cache`)해서 모든 Experimenter/Trainer가 공유
- 키는 **`(scope, node, typ)`** — `scope`는 그 fold를 만든 `TrainDataFlow`가 **자기 생성자에서 만드는 랜덤 id**(`self.scope = uuid.uuid4().hex`). 경로 문자열을 쓰지 않는 이유: run을 Project 없이 독립 생성하고 `cache=`를 외부 주입받을 수 있으므로, 서로 다른 물리 디렉토리가 우연히 같은 상대경로 문자열이 되어 충돌할 수 있음
  - **`TrainDataFlow` 인스턴스 하나 = 정확히 그 (run, fold) 하나**(fold마다 새로 만들고 공유 안 함)라, 인스턴스 자신의 랜덤 id만으로 이미 "이 run의 이 fold"가 유일하게 식별됨 — `outer_idx`/`inner_idx`를 키에 또 넣지 않는 이유
  - 트레이드오프: 리로드하면(새 Python 인스턴스 = 새 scope id) 이전 인스턴스가 캐싱해둔 항목은 다시 못 만남 — cache miss일 뿐 잘못된 값이 나오는 게 아니라 허용 가능한 손실로 판단
- `get_data(scope, node, typ)`, `put_data(scope, node, typ, data)`
- `clear_nodes(nodes)`: 특정 노드들의 캐시 삭제(이름만 매칭 — scope 무관하게 지움. 여러 run이 같은 노드 이름을 쓰면 서로의 캐시까지 같이 지워짐. 안전하지만 낭비 — 미해결)

### NodeStore (`_store.py`)
- `ArtifactStore`를 상속해 아티팩트 메소드를 전부 구현
- **run 하나(Experimenter/Trainer 하나)당 인스턴스 하나** — `outer_idx`/`inner_idx`를 매 호출마다 받아서 그 fold의 경로/이력을 그때그때 계산. Experimenter/Trainer가 각자 자기 base path(`exp/{name}`, `trainers/{name}`)로 생성자에서 한 번만 만들고, 그 run의 모든 fold가 같은 인스턴스를 공유
- 아티팩트: `{path}/{outer_idx}/{inner_idx}/{node_name}/`
  - `obj.pkl` — processor 객체, `result.pkl` — fit_transform/fit_predict 출력. **info.pkl 없음** — status/definition/fit_time/edges/train_shape/error는 전부 history에
  - `node_path(name, outer_idx, inner_idx)`가 경로 조립. `write_objs`/`write_obj`/`write_result`/`get_objs`/`get_obj`/`get_result`/`list_nodes`/`status`/`reset_node` 전부 fold를 인자로 받는 instance 메소드
  - `status(name, outer_idx, inner_idx)`는 `None`/`'built'`만 반환(obj.pkl 존재만 봄) — `'error'`는 여기서 절대 안 보임
  - `NodeStore`는 열린 커넥션을 `self`에 들고 있지 않아 picklable — 서브프로세스 워커에 스폰 시점에 인스턴스 자체를 넘긴다(`ProcessWorker(conn, collectors, store, ...)`)
- **history**: SQLite `node_hist(node_name, outer_idx, inner_idx, pipeline_version, status, info)` — PK는 `(node_name, outer_idx, inner_idx)`(run_name 컬럼 없음 — store 자체가 이미 그 run에 스코프돼 있음)
  - `record(name, outer_idx, inner_idx, pipeline_version=, status=, info=)` — `info`는 `status` 제외 나머지 전부(`build_id`, `definition`, `fit_time`, `edges`, `train_shape`, `warnings`, 실패 시 `error`) JSON 인코딩. `NodeInfoTracker`(`_tracker.py`)가 기록
  - `get_hist(node_name=, outer_idx=, inner_idx=, pipeline_version=)`/`get_status(node_name)`/`get_info(node_name)`(`{(outer_idx, inner_idx): ...}`)/`get_fold_info(outer_idx, inner_idx)`(fold 하나의 `{node_name: info}` 전체 — `DataFlow.load()`가 씀)/`remove_hist(node_name=)`
  - `'error'`는 오직 여기서만 보임 — `Experimenter.show_error_nodes`/`Trainer.get_node_error`가 조회

### DataFlow / TrainDataFlow (`_flow.py`)
- **DataFlow**: 생성자가 `NodeStore` 인스턴스(`self.store`, run 전체가 공유) + 이 fold의 `outer_idx`/`inner_idx`를 받음. `status`/`get_obj`/`get_objs`/`get_result`/`list_nodes`/`node_path`는 `self.store.X(name, self.outer_idx, self.inner_idx)`로 위임하는 얇은 메소드. `reset_node`만 위임 + `node_objs`/`_node_edges`에서도 같이 지우는 조합 동작
  - `node_objs`: `{name: (obj, result)}`, `_node_edges`: `{name: edges}`
  - **`load()`가 `self.store.get_fold_info(...)`를 한 번 조회**해서 `edges`까지 복원한 뒤 `load_objs(name, edges=...)`. history에 행이 없는 노드는 로드 안 함(안전한 기본값). 여기 올라오는 건 Pipeline 노드뿐이다 — Trial은 애초에 아무것도 안 남기고 Trainer의 Predictor는 자기 store에 있어서, 둘 다 이 store에 아티팩트를 두지 않는다(학습된 모델이 메모리로 딸려 들어오지 않음)
  - `get_data(source_data, edges)` → `{key: data}`
- **TrainDataFlow** (DataFlow 상속): 노드 빌드 기능 추가. `store`를 그대로 받아 `super().__init__(store, outer_idx=, inner_idx=)`로 넘김 — fold별로 자기 NodeStore를 새로 만들지 않음
  - `data_source`: DataWrapperProvider (train/valid/**test** 제공 — `test_idx` 보유)
  - `outer_idx`/`inner_idx`는 NodeStore 키(아티팩트 경로, history row) 전용 — DataCache 키에는 안 들어감(`self.scope`가 대신함). **Trainer도 자연스러운 `(split_idx, 0)`을 그대로 쓴다**
  - `get_train(edges)`, `get_valid(edges)`, `get_test(edges)` — flow 하나로 job의 모든 입력을 만들 수 있어야 `Job`이 자족적이 됨
  - `set_objs(name, obj, result, info)`: 현재 fit의 즉석 info에서 `edges`만 추출 — 디스크/history를 안 거침. **`chained` 실행(=노드 빌드)에서만 호출**된다 — leaf(Trial/Predictor)를 여기 올리면 아무도 안 읽는 모델이 flow 메모리에 남는다

### Trainer (`_trainer.py`)
- **Project 의존성 없음. 주입받는 건 `cache` 하나**(Experimenter와 동형) — 생성자: `Trainer(path, name, data, splitter=None, splitter_params=None, aug_data=None, cache=None)`. store(`TrainerStore(self.path)`)는 자기가 만들고, Pipeline은 `set_pipeline()`으로만
  - `data`는 native/`DataWrapper` 아무거나 — 생성자와 `load_trainer()` 둘 다 `wrap(data)`
  - **생성자 = 신규 생성**, 복원은 **`Trainer.load_trainer(path, data, aug_data=None, cache=None)`** staticmethod. `{path}/__trainer.db`가 없으면 `KeyError`이고, Experimenter와 같은 이유로 **store를 만들기 전에** 검사
  - 복원 시 split은 **재계산이 아니라 저장된 `split_indices`를 그대로** 씀 — splitter가 아예 없는 Trainer(단일 full-data fold)도 있고, 학습된 fold와 정확히 같아야 하므로
- 경로 `{project}/trainers/{name}`
- **`set_pipeline(pipeline, pipeline_name=None)`**: 이미 로드된 `Pipeline` 객체를 받음 — `self.pipeline_version`은 `pipeline.version`에서 읽음. 버전 전환 시 `diff_from`으로 stale 제거
- **저장소가 넷**:
  - `_store`: `TrainerStore({path})` — meta + splits BLOB + `pipeline.pkl`
  - `node_store`: `NodeStore({path})` — Pipeline 노드 아티팩트+이력
  - `predictor_store`: `NodeStore({path}/__predictors)` — Predictor 아티팩트+이력. 같은 클래스, 다른 디렉토리
  - `predictor_defs`: `PredictorStore({path}/__predictors)` — Predictor **정의**
  - **디렉토리 분리는 강제** — 두 `NodeStore`가 한 디렉토리를 쓰면 `__node_hist.db` 파일명이 충돌. 덤으로 노드/Predictor가 디스크에서 구조적으로 갈림
  - 이 구조 덕에 Predictor 이력용 run 이름 컬럼도, `exp/{name}`↔`trainers/{name}` 네임스페이스 충돌도, 전용 tracker도 필요 없다 — store가 이미 run 스코프라 `NodeInfoTracker`를 그대로 씀
- **선택 단계(`set_predictors()` 류)는 없다** — `train(predictors)`가 직접 받는다
- **`predictors`/`selected_nodes`는 property** — `predictors`는 `predictor_defs.list_predictors()`(상태로 안 들고 있음 → 리로드가 공짜), `selected_nodes`는 `_nodes_for(self.predictors)`로 매번 계산(Predictor가 없으면 전 노드)
- `predictor_names()`, `predictor_specs()`
- `train_folds`: `[TrainFold]` — split당 `TrainDataFlow` 하나
- **`train(predictors=None, n_jobs=1, gpu_id_list=None, logger=None)`**: 노드 먼저(위상 순서), 그 다음 Predictor `Job` 실행. 두 실행은 별개 executor 호출이라 각자 자기 store와 자기 `NodeInfoTracker`를 받음. skip 판정은 양쪽 다 디스크 기반(`store.status(name, split_idx, 0) == 'built'`) — **재정의는 그 자체로 재학습을 유발하지 않는다**(강제하려면 `reset_nodes([name])`)
  - `predictors`는 `predictor_defs.register_all()`로 **upsert 등록**(replace 아님) — 이전 호출에서 학습한 Predictor의 정의와 아티팩트가 살아남는다. replace로 지우면 아티팩트만 남고 정의가 사라져 읽을 수 없게 됨
  - `predictors=None`이면 이미 등록된 것들을 그대로 이어서 학습(중단된 학습 재개)
  - `Trial`을 넘기면 `TypeError` — `Predictor.from_trial(trial, experimenter=)`로 명시 승격해야 출처가 기록됨
  - 학습 대상 노드는 **넘긴 predictors 기준**(`_nodes_for(predictors)`)이지 등록된 전체가 아님
  - `Job.flow`는 Predictor에도 **노드 flow**가 들어감(입력을 만드는 건 노드 그래프) — 아티팩트는 `predictor_store`로 가고, **flow에는 게시되지 않는다**(`chained=False`). Predictor는 leaf라 아무도 안 읽는데 flow에 올리면 모델만 메모리에 남기 때문. 나중에 필요한 쪽(`process()`/`to_inferencer()`)이 `predictor_store`에서 직접 꺼낸다
- `get_status(node_name)` / `get_node_error(node_name)`: `_store_for(name)`으로 두 store 중 하나를 골라 조회 — Predictor 에러도 기록됨
- `process(data, v=None)`: generator, split마다 Predictor output을 `v`(DSL 문자열)로 필터 후 concat하여 yield. Predictor 모델은 `flow.load()`가 안 집어오므로(이력이 다른 store에 있음 — 그게 메모리에 안 딸려오게 하는 장치) `predictor_store`에서 on-demand로 꺼내고, **edges는 정의에서 채운다** — 아티팩트에도 이 flow의 이력에도 없기 때문(안 채우면 `_resolve`가 전부 None을 반환해 조용히 아무것도 yield 안 함)
- `to_inferencer(v=None)`: 학습된 Processor를 추출하여 Inferencer 생성
- `reset_nodes(nodes)`: 하위 종속 노드 포함 초기화. Predictor는 leaf라 그래프 캐스케이드 대상이 아니지만, 리셋된 노드를 읽는 Predictor는 `node_names()` 교집합으로 같이 리셋됨
- 저장/로드: `save()`(meta + splits를 `_store`에 기록), 복원은 `Trainer.load_trainer()`. Pipeline은 `{path}/pipeline.pkl`, splitter/split_indices는 `__trainer.db`의 splits BLOB, **Predictor는 `predictor_defs`에서** 복원

### Predictor (`_predictor.py`)
Trainer가 학습하는 끝지점 출력 노드. `name`, `processor`, `method`, `adapter`, `params`, `edges`, `desc`, `tag`, `src_trial`, `src_experimenter` (`__slots__`)

- **Trial과 왜 별도 클래스인가**: 둘 다 `get_spec()`이 구분 불가능한 `ProcessorSpec`을 내놓으므로 구조만으로는 나눌 이유가 없다. 나누는 건 **의미**다 — Trial은 *비교되는 후보*라서 `TrialStore`가 프로젝트 전역이고 모든 Experimenter의 fold별 결과를 모으는 반면, Predictor는 *이미 내려진 결정*이라 중요한 게 비교 가능성이 아니라 **출처**(어떤 후보가 이걸 정당화했는가). 그래서 `src_trial`/`src_experimenter`를 들고 레지스트리는 Trainer별
- `Predictor.from_trial(trial, name=None, experimenter=None)`: 실행 정의를 그대로 복사하고 Trial 이름을 `src_trial`로 기록(`name`으로 이름을 바꿔도 출처는 원래 이름으로 남음)
- `get_spec()` → `ProcessorSpec`(Trial/노드와 동일), `node_names()`: edges가 참조하는 노드 이름 집합

### PredictorStore (`_predictor_store.py`)
```sql
predictors(name PK, desc, processor, method, adapter, params, edges, tag,
           src_trial, src_experimenter)
```
- **정의만 있고 이력 테이블이 없다** — Predictor의 fold별 status/info는 그 Trainer의 `predictor_store`(`NodeStore`)의 `node_hist`에 아티팩트와 같이 있음. `TrialStore`가 두 절반을 다 갖는 것과 대비되는데, TrialStore는 프로젝트 전역이라 "어느 Experimenter가 돌렸나"를 답해야 하는 반면 여기는 store 자체가 이미 그 Trainer로 스코프돼 있어서
- **Trainer별인 이유**: Trial은 **결과가 중심**이라 여러 Experimenter의 성적을 한곳에 모아 비교해야 한다. Predictor는 **아티팩트가 중심**이고 서로 비교할 일이 없다 — 답하는 질문이 "*이* Trainer가 뭘 학습하나"라서 프로젝트 레벨로 공유할 이유가 없음. `Trainer`가 `self.predictors`를 상태로 안 들고 이 store를 직접 읽는 property로 둔 근거이기도 함
- `register`/`register_all`(upsert — `Trainer.train`이 쓰는 것)/`replace_all(predictors)`(통째 교체, 빠진 이름 삭제)/`remove(name)`/`has(predictor)`/`get_by_name(name)`/`list_predictors()`
- `get_by_name`/`list_predictors`는 dict가 아니라 **`Predictor` 객체**를 반환(`TrialStore`와 다름) — `Trainer.predictors` property가 그대로 내놓기 때문
- `ArtifactStore`를 상속하지 않음 — 아티팩트도 이력도 안 갖는 순수 정의 레지스트리

### Inferencer (`_inferencer.py`)
- 생성자: `(node_specs, selected_nodes, selected_predictors, n_splits, node_objs, v=None)`
- **Pipeline 의존성 없음** — `node_specs`(`{name: ProcessorSpec}`)만 보유. 실제로 필요한 건 `spec.edges`뿐이라 배포 아티팩트가 가볍다
- `node_objs`: `{name: [processor_split0, processor_split1, ...]}` — Processor 리스트 (Trainer 독립)
- `process(data, agg='mean', nodes=None)`: split 결과 자동 집계 (`nodes`는 Predictor 이름 필터)
  - `agg`: `'mean'`/`'mode'`/callable/`None`(list 반환). 단일 split이면 집계 없이 반환
  - `nodes`: str/list — 출력할 노드 선택 (None=전체). 미등록 노드 지정 시 ValueError
- 저장/로드: `save(path)`, `load(cls, path)` — 단일 `__inferencer.pkl`에 node_objs 포함

### Connector (`_connector.py`)
- `__init__(node_query=None, edges=None, processor=None)` — 3요소 선택적 매칭. `role` 같은 노드/Trial 구분 파라미터는 없다(Collector가 Trial job에만 붙으므로 걸러야 할 대상 자체가 없음)
- `processor`: **`"module.ClassName"` 문자열만** (클래스 인스턴스 아님) — resolve 안 하고 그대로 저장
- `match(spec)`: `ProcessorSpec`을 받아 설정된 요소만 검사, 모두 충족 시 True
  - node_query: str(regex) 또는 list(in)
  - processor: `spec.processor`(문자열)와 **문자열 그대로 비교**(정규화 없음 — `set_grp`/`set_node`에 준 것과 같은 문자열 형태를 넘겨야 매칭)
  - edges: `{key: dsl_string}` — 각 key에 대해 노드의 resolved `edges[key]` 문자열과 **정확히 일치**해야 함 (contain 기반 아님)

### Collector (`collector/` 패키지)
- **Collectors** (`_registry.py`): Collector 인스턴스를 소유하는 레지스트리. **Experimenter 하나당 하나** — `e.collectors`(`{exp path}/collectors`)
  - `Collectors(path=None)` — path 있으면 등록 시 `{path}/{name}`이 기본 저장 위치. **생성자가 곧 복원** — 그 path의 `CollectorStore`에서 등록돼 있던 걸 전부 되살림(별도 `load()` 없음)
  - **왜 run 소유인가**: Collector가 쓰는 건 전부 노드 이름만으로 키잉된다(`MetricCollector` PK `(node, idx, inner_idx, split)`, 파일 기반은 `{path}/{node}...`). 그래서 두 run을 가르는 건 **경로뿐**이고, 레지스트리를 공유하면 이름이 겹치는 Trial마다 서로의 결과를 덮어쓴다 — 무증상으로, 그리고 하필 비교가 필요한 바로 그 경우에. `NodeStore`와 같은 배치이고, 덕분에 standalone run이 collector까지 자족한다
  - `set_collector(name, collector, connector, path=None, params=None, exist='skip')` — 부품에서 조립. `collector`는 클래스 또는 `"module.ClassName"`, `connector`는 인스턴스 또는 `{__ref__}`, `params`엔 `resolve_ref_values` 적용
  - **등록 즉시 영속화** — `set_collector`가 store에 write-through. `Collectors.save()`는 없다. path 없는 레지스트리는 메모리 전용
  - `get_collector`/`remove_collector`(store 행+params 파일까지 삭제)/`names()`/`in`/`len`/`iter`
  - **`resolve(names)`**: 미등록 이름이면 `KeyError` — 조용히 넘어가면 "아무것도 수집 안 됨"과 구분이 안 되기 때문. `None`이면 전부
  - **`remove_results(node_name)`**: 그 노드로 수집된 것 전부 삭제 — `hist` 행 + 각 Collector의 `reset_nodes([name])`. 두 절반을 다 보는 게 레지스트리뿐이라 여기 있다(정의는 안 건드림). `Experimenter.remove_trial_result`/`Project.remove_trial`이 이걸로 수렴
  - `match(spec, names=None)`
  - **`hist`**: 같은 path의 `CollectHist`. path 없으면 `None`

- **CollectHist** (`collector/_collect_hist.py`): 수집 이력
  ```sql
  collect_hist(collector_name, node_name, outer_idx, inner_idx,  -- PK
               pipeline_version, status, collect_date, elapsed, info)
  ```
  - **`experimenter` 컬럼이 없는 이유**: 레지스트리가 이미 run 하나에 스코프돼 있어 어느 run인지는 **db가 어디 있느냐**로 답한다 — `node_hist`에 `run_name`이 없는 것과 같다
  - **키가 fold 단위인 이유**: Collector가 자기 상태를 다루는 단위는 노드(`has_node`/`abort_node`/`reset_nodes`/`_save_node`)지만, 실패는 fold 하나에서 나고 그게 어느 fold였는지가 분석의 시작점. 노드로 접으면 그 정보가 사라짐
  - `status`: `'collected'`(결과 반환) / `'empty'`(예외 없이 `None`) / `'error'`. **`empty`를 따로 가르는 게 핵심** — 실패와 "수집할 게 없음"이 둘 다 `None`이면 구분이 안 된다(보통 `output_var` 설정 실수)
  - `info`(에러 시 JSON): `{phase, type, message, traceback}`. `phase`는 `'output'`(공용 `obj.process` 준비) / `'ext'`(ProcessCollector의 `output_ext`) / `'collect'` / `'push'`
  - `elapsed`는 **`collect()` 호출 시간만** — 단일/멀티 워커에서 같은 의미가 되도록(멀티에선 push가 부모에서 돎)
  - `record`/`record_all`/`get_hist(collector_name=, node_name=, status=, pipeline_version=)`/`get_status(collector_name, node_name=)`/`get_info`/`get_errors(collector_name=)`/`remove_hist(collector_name=, node_name=)`
  - `get_status`/`get_info`의 키는 `(node_name, outer_idx, inner_idx)`
  - **게이트가 아니라 로그** — `experiment_hist`와 달리 이걸 보고 뭘 스킵하지 않는다. 다만 "`'built'`로 스킵된 fold엔 수집 기록이 없다"가 조회로 드러나서, 이미 다 돌린 실험에 collector를 새로 붙여 `exp()`를 다시 불렀을 때 아무것도 수집 안 되는 상황이 무증상으로 지나가지 않는다

- **CollectorStore / CollectorEntity** (`collector/_store.py`): Collector 정의 저장소
  - `collectors(name PK, collector TEXT, connector TEXT, path TEXT)` + `{path}/__params/{name}.pkl`
  - **인스턴스를 저장하지 않는다** — 조립 부품 두 쪽(entity 행 + params pkl)만 남기고 로드 때 `build_collector(entity, params)`로 **다시 조립**. `set_collector`와 완전히 같은 경로를 타므로 등록과 복원이 구조적으로 같아진다. 실행 중 인스턴스에 붙는 값(`_n_outer`/`_n_inner`)은 영속화 대상이 아님
  - `params`가 **pkl 파일**인 이유: `ProcessCollector(ext_data=df)`처럼 정의로 표현 불가능한 산 객체가 들어옴 — 노드/Trial처럼 JSON 강제를 할 수 없어서 이 한 조각만 pickle. 나머지 4개는 평문 컬럼이라 **unpickle 없이** 목록/내용 조회 가능(`list_entities()`)
  - `CollectorEntity`(`__slots__`: `name`/`collector`/`connector`/`path`) — 한 행의 표현. `of(...)`가 준 대로의 **문자 원형**으로 정규화(클래스면 `_obj_to_ref`, `Connector` 인스턴스면 `{__ref__, __params__}`)
  - `register(entity, params)` / `build(name)` / `load_all()` / `get_entity` / `list_entities` / `get_params` / `names` / `remove`
  - **Collector 클래스는 모듈 최상위여야 한다** — 함수 안에서 정의한 서브클래스는 ref로 resolve할 수 없고, 멀티워커 실행이 collector를 pickle해 워커로 보내므로 어차피 필요한 조건. `Collector.__getstate__`/`_SAVE_EXCLUDE`가 그 경로를 위해 존재

- **Collector** (`_base.py`): 기본 클래스
  - `__init__(name, connector)`, `path`는 `Collectors.set_collector` 시 설정
  - 라이프사이클: `collect(context)` → `push(node, outer_idx, inner_idx, result)` → (inner 버퍼가 차면) `_flush_outer(node, outer_idx, inner_list)` → (서브클래스에 따라) `_save_node(node)`
  - **에러 처리**: `_executor._safe_collect`가 `ext`/`collect`/`push` 세 구간을 전부 try/except로 감싸 `CollectHist`에 기록하고 실행을 계속한다. 이력은 항상 부모 프로세스가 쓰므로(`TrialHistTracker`) 단일/멀티 두 모드가 같은 경로. Collector 인스턴스에 경고를 쌓는 필드는 없다(워커로 pickle돼 나가면 거기서 버려지므로 성립하지 않음)
  - `push`를 감싸는 건 기록 목적만이 아니다 — 멀티워커에선 부모 메시지 루프가 `c.push()`를 부르므로, 여기서 예외가 나면 워커에 종료 sentinel이 못 가서 좀비 프로세스가 남는다
  - `on_attach(experimenter)`: `exp()`가 호출 — experimenter identity 비교로 중복 재계산 방지; `_on_attach(experimenter)` no-op 훅을 subclass에서 override
  - `_experimenter`: pickle 제외 (`__getstate__`/`__setstate__` — 워커로 보낼 때 None으로 초기화)
  - `has_node(node)`: 수집 결과 보유 여부
  - `reset_nodes(nodes)`: base는 `self._buf`에서 해당 노드 제거 — 서브클래스는 `super().reset_nodes(nodes)` 먼저 호출 후 자신의 disk/cache 정리
  - `save()`/`load()`는 없다 — 정의는 `CollectorStore`가, 데이터는 각 서브클래스가 자기 `path`에 이미 즉시 기록한다
  - `_get_nodes(nodes, available)`: None/list/str(regex) 패턴 매칭
  - context: `{node_spec, processor, info, input, outer_idx, inner_idx, output_train, output_valid, output_test, output_ext}` — `node_spec`은 `ProcessorSpec`, `info`는 `_process()`의 info dict

- **MetricCollector** (`_metric.py`): 메트릭 수집
  - `output_var`(DSL 문자열 또는 None), `metric_func`, `include_train`
  - target: `context['input']['y']`, 예측값: `eval_expr(parse(output_var), output_valid, processor=context['processor'])`로 컬럼 선택
  - `_on_attach`: `metric_func`에 `on_attach`가 있으면 자동 전파
  - 저장: `push()` 오버라이드로 inner fold 결과 발생 시 즉시 `metrics.db` INSERT
  - 쿼리: `get_metric(node)`, `get_metrics(nodes)`, `get_metrics_agg(nodes, inner_fold, outer_fold, include_std)`

- **ProbToLabel** (`_metric.py`): predict_proba → label 변환 후 metric 적용
  - `__init__(metric_func, var, thresholds=None)` — `metric_func`를 래핑하는 callable class
  - `var`: DSL 문자열 (예: `'{target}'`) — `on_attach`에서 `experimenter.get_test_data({'_y': var})`로 resolve
  - `thresholds`: None=argmax, float=binary threshold, list=multiclass per-class threshold
  - `on_attach`에서 experimenter로부터 label classes 추출 (정렬 순서 = predict_proba 열 순서)
  - binary: 2D proba `(n, 2)` 자동 처리 (col 1 추출), 1D sigmoid도 지원
  - multiclass per-class threshold: threshold 초과 클래스 중 최대 확률 선택, 없으면 argmax fallback

- **StackingCollector** (`_stacking.py`): 스태킹 데이터 수집
  - `__init__(name, connector, output_var, method='mean')` — experimenter 불필요, `_on_attach` 없음
  - **집계는 읽기 시점으로 미룬다** — `_flush_outer`는 inner 결과를 집계하지 않고 **리스트 그대로** `_outer_buf`에 쌓고, `_save_node`가 `{node}.pkl`에 `{'folds': [[inner...], ...]}`로 저장. `get_dataset(experimenter, ...)`가 experimenter에서 `data_cls`/`n_splits`/index/target을 얻어 그때 집계·결합(index/target은 설정이 아니라 데이터셋 크기의 배열이라 인스턴스에 들고 있지 않음)
  - `_aggregate(data_cls, iterator)`: `method`(mean/mode/simple)에 따라 `data_cls`의 static 메서드 호출
  - 쿼리: **`get_dataset(experimenter, nodes=None, include_target=True)`** — experimenter가 필수 인자. 저장된 fold 수가 그 experimenter의 `get_n_splits()`와 다르면 `ValueError`

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
  - `ext_data`는 미저장 (런타임 전달)

## edges 구조 — DSL 문자열 (`_edge_dsl.py`)
- dict 형태: `{key: dsl_string}` — key는 변수 집합 이름(예: 'X', 'y', 'sample_weight'), 값은 **항상 순수 문자열**
- `edges[key]`는 정의/상속/직렬화/비교 어디서나 문자열 그대로 유지되며, **Processor 실행 시점에 실제 데이터를 대상으로만** lazily 컬럼 리스트로 확장됨(`eval_expr`, `_flow.py.get_data`). `set_grp`/`set_node`는 DSL *구조*(문법 + namespace 참조)만 검증(`validate_edges`)하고 컬럼/schema는 절대 만들어보지 않음
- 문법:
  ```
  expr        := term (op term)*            -- op는 '+'/'-'/'&', 공백으로 양쪽이 분리된 독립 토큰이어야 함
  term        := ('*' | set_literal | pattern) ['@' NAME ['(' ')']]
                  | slice | namespace | '(' expr ')'
  slice       := [INT] ':' [INT]             -- 파이썬 slice, 예: '-1:' == slice(-1, None)
  set_literal := '{' [NAME (',' NAME)*] '}'  -- 명시적 컬럼명 목록
  namespace   := NAME ':' '(' expr ')'       -- NAME은 노드명, 그 노드의 출력을 가리킴
  pattern     := REGEX                       -- re.match 로 컬럼명 매칭
  ```
  - `*`/`set_literal`/`pattern` 뒤에 바로(공백 없이) `@name`/`@name()`을 붙이면 `col.py`에 등록된 column-selector 적용 (예: `*@numeric`, `{a, b}@int`, `A.*@ohe_drop_first`)
  - 값을 직접 명시하지 않은 top-level(namespace 밖) 항목은 DataSource를 가리킴; `name:(...)` 블록은 그 노드의 출력을 가리킴 (namespace는 top-level에서 `+`로만 결합 가능 — `-`/`&`는 namespace 내부/괄호 안에서만)
  - **DataSource 참조는 반드시 명시적 컬럼명 리스트(`{a, b}`)** — 패턴/callable 아님. [[feedback_datasource_edges_explicit_vars]]
- 그룹/노드 상속: 자기 값이 `+`/`-`로 시작하면 부모의 이미 resolve된 문자열에 이어붙임(`f"{parent} {own}"`); 그 외 일반 문자열은 완전히 override(상속 안 함). 자기 값이 없으면(`{}`) 부모 값을 그대로 상속
- 같은 key의 여러 segment(`+`로 연결)는 column 방향으로 concat됨

### Edge DSL 관련 함수 (`_edge_dsl.py`)
- `parse(dsl_string)` → AST (`Star`/`SetLiteral`/`Pattern`/`Namespace`/`BinOp`/`slice`)
- `eval_expr(node, data, processor=None)`: AST를 실제 `data`(`DataWrapper`, `get_columns()`/`select_by_dtype()` 노출)에 대해 평가 → 컬럼명 리스트. `data`에서 `columns`를 내부적으로 유도하므로 호출부는 컬럼 리스트를 따로 넘기지 않음
- `validate_edges(dsl_string, pipeline)`: 구조만 검증(문법 + namespace가 존재하는 노드를 가리키는지) — 컬럼/schema는 절대 건드리지 않음
- `iter_segments(dsl_string)` → `(node_name, expr)` 이터레이터 (top-level `+` 체인 분해)
- `referenced_nodes(dsl_string)` → 참조하는 노드명 집합 (`None`=DataSource 포함)
- `unparse(node)` → AST를 다시 DSL 문자열로 렌더링

## col.py — `@name` column-selector 레지스트리
- `col_selector(*processor_classes, name=None)` 데코레이터로 등록. 모든 selector는 동일 시그니처 `(data, processor=None) -> mask` — `data`는 이미 패턴 등으로 좁혀진 후보 컬럼만 담은 `DataWrapper`
- `processor_classes`가 지정되면 해당 processor 타입에서만 유효(`resolve_selector`가 불일치 시 ValueError); 비워두면(`()`) processor 없이도 사용 가능
- `name=`으로 등록 키를 함수명과 다르게 지정 가능 (파이썬 builtin과 겹치는 `float`/`int`/`string` 등에 사용)
- **`ohe_drop_first`** (OneHotEncoder 전용), **`subset_poly`** (PolynomialFeatures 전용, degree/interaction/bias 일관된 전체 조합으로 스냅)
- **dtype 기반 builtin selector** (processor 불필요, `data.select_by_dtype(kind)` 사용): `@numeric`, `@categorical`, `@binary`(bool dtype만), `@float`, `@int`, `@string`
  - DataSource 최상위(`*@numeric` 등)에 바로 걸면 schema에 없는 raw 컬럼(id/target/sample_weight 등)까지 포함될 수 있어 위험 — 이미 확정된 노드 출력 namespace 안에서 쓰는 게 안전

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
정의 변경 플래그(serial 류)가 아니라 **버전 간 구조 비교**로 판정한다. 판정 지점은 `set_pipeline()` **한 곳**이고, `build()`/`train()`은 "디스크에 있는 건 유효하다"를 전제할 수 있다. 이 방식의 부산물로 **자기가 읽지 않는 노드를 고쳐도 Trial이 유지된다**(전역 플래그로는 이 구분이 안 됨).

**`Pipeline.diff_from(old)`** — DataSource에서 위상 순서로 내려가며:
1. 이름이 old에 없으면 → stale
2. 정의(`_definition_of` = `processor/method/adapter/params/edges`)가 다르면 → stale
3. 정의는 같아도 **읽는 노드 중 하나가 stale이면** → stale (하위 전파가 위상 순서에서 자동으로 나옴)
4. old에는 있는데 지금 없는 이름 → stale (아티팩트 청소용)
5. DataSource schema/targets가 바뀌면 → 전부 stale

- **Trial은 캐스케이드하지 않는다**: Trial은 Pipeline 밖이라 diff가 이름을 모르고, stale 노드를 읽은 Trial도 건드리지 않는다 — `Experimenter.set_pipeline`은 stale 노드만 `reset_nodes`로 지우고 끝. Trial의 결과가 어떤 pipeline 버전에 대한 것인지는 `TrialStore.experiment_hist`가 `pipeline_version`으로 기록하므로 그 버전에 대한 기록으로 남고(버전을 올릴 의도가 없으면 Pipeline이 바뀔 이유도 없다), 다시 돌리려면 `remove_hist`로 명시적으로 재실행해야 함
- **Trial 자신의 재정의도 감지하지 않는다**: `Experimenter._make_jobs`는 오직 `experiment_hist`의 fold별 status만 본다 — 재정의된 trial이 이미 `'built'`면 조용히 스킵되므로, 다시 돌리려면 `reset_nodes([trial_name])`을 직접 호출
- **Predictor는 반대로 캐스케이드한다**: Trainer엔 보존할 "과거 실행" 개념이 없으므로, 바뀐 노드를 읽는 Predictor는 그냥 stale이다. `Trainer.reset_nodes`가 `predictor.node_names() & 리셋된_노드` 교집합으로 같이 지운다. Trial/Predictor를 별도 클래스로 둔 판단이 실제로 갈라지는 지점

## Lazy resolution: processor / adapter / params (edges DSL과 동일한 지연 원칙)
- `set_grp`/`set_node`는 `processor`/`adapter`/`params`를 **스펙 형태만 검증하고 절대 resolve/instantiate하지 않는다** — 실제 값으로의 변환은 전부 사용 시점(`_node_processor.py`)으로 미룸
- **산 객체는 정의 시점에 `TypeError`로 거부된다** (`_validate_processor`/`_validate_adapter`/`_validate_params`, `_pipeline.py`). 에러 메시지가 써야 할 ref 형태를 그대로 안내함
- **processor**: `"module.ClassName"` **문자열만** (클래스 객체 거부). 실제 클래스로 resolve되는 유일한 지점은 `_node_processor.py`의 `resolve_processor()` 호출. `Connector`, `_describer.py`, `Experimenter._make_node_jobs`/`Trainer._make_node_jobs`(GPU 판정)는 전부 이 문자열을 그대로 다룸
  - **클래스를 허용하면 안 되는 이유**: `Connector.match`는 `spec.processor`를 **문자열 그대로 비교**하므로, 클래스로 정의된 노드는 문자열 ref로 설정한 Connector와 **영영 매칭되지 않는다** — 에러 없이 collector가 조용히 아무것도 수집하지 않음. `serialize_value`가 클래스를 `{"__type__":"class"}`로 저장하고 리로드 시 클래스로 되돌리므로 재시작해도 해소되지 않음
- **adapter**: `None` / `"module.ClassName"` 문자열 / `{"__ref__":...,"__params__":{...}}` dict — 인스턴스 거부. `resolve_node_adapter(processor, adapter_spec)`(`adapter/__init__.py`)가 사용 시점에 resolve
- **params**: 순수 데이터(스칼라/numpy 스칼라/list/tuple/dict)와 ref spec만 허용 — `{'__ref__':...}`(예: ColSelector, `mllab_sampler`)/`{'__callable__':...}`(예: metric 함수). 중첩 값까지 재귀 검증하며 에러가 경로(`params['a']['b'][0]`)를 표시함. `_node_processor.py`의 `_resolve_params()`가 Processor 생성 시점에 `resolve_ref_values()`로 해제
- **왜**: (1) `mllabs.nn.NNClassifier`처럼 무거운 import(TensorFlow)를 유발하는 processor/adapter가 파이프라인 "정의" 시점에 로드되는 걸 방지 — 실제 `build`/`exp` 실행 시점까지 미뤄짐. (2) 파이프라인 전체가 직렬화 가능해짐(declarative config 방향). (3) diff 비교가 raw spec 비교라 `_params_equal`이 `==` 한 줄로 끝남 — 인스턴스 `__eq__` 신뢰성 이슈가 없음
- **테스트**: string-ref로 참조돼야 하는 더미 processor 클래스는 `tests/mock.py`에 모아둔다 — `tests/`엔 `__init__.py`가 없어 pytest가 bare module(`import mock`)로 수집하므로 `processor='mock.DummyStage'`식으로 참조 가능

## Processor (`_node_processor.py`)
- **TransformProcessor**: `fit`, `fit_process`, `process`
- **PredictProcessor**: `fit`, `fit_process`, `process`
- **processor(=transformer/estimator)/adapter/params가 실제 값으로 resolve되는 유일한 지점** — Pipeline은 이 셋을 절대 resolve 안 하고 spec 그대로 넘김("Lazy resolution" 섹션)
  - `__init__`: `self.transformer`/`self.estimator = resolve_processor(...)` — processor가 클래스가 되는 유일한 곳
  - `self.adapter = resolve_node_adapter(transformer/estimator, adapter)` — **resolve 전의 raw(문자열) processor**를 넘김("processor는 인스턴스 생성 시점에만 클래스로" 원칙)
  - `self.params = _resolve_params(params)` — `{'__ref__':...}`/`{'__callable__':...}` 항목을 `resolve_ref_values()`로 해제(ColSelector 인스턴스화 등). `mllab_sampler` 값도 여기서 같이 resolve
- `fit`/`fit_process`에서 y 데이터를 `squeeze()` 후 전달 (sklearn DataConversionWarning 억제)
- `get_feature_names_out` 반환값은 `list()` 로 변환하여 사용 (list/ndarray 호환)
- `process()`: `adapter.get_process_data(data)` 로 입력 타입 변환 — polars 등 라이브러리별 호환성 처리
- `data_dict` (Experimenter): `{key: ((train, train_v), valid), ...}` / (Trainer): `{key: (train, valid), ...}` (inner fold 없음)
- **X-less 지원**: `edges`에 `'X'`가 없고 `'y'`만 있는 경우(e.g. `LabelEncoder`) `'y'`를 primary input으로 사용
  - `fit`/`fit_process`: `'X'` 없으면 `'y'` 데이터를 squeeze하여 전달, `output_vars`를 `y_columns`로 설정
  - `process`: `X_`가 비어 있으면 입력 데이터를 squeeze 후 transform
- `y_columns`가 str인 경우(polars Series 등) `[y_columns]` 로 wrap하여 처리

### TransformProcessor 출력 컬럼 규칙
- transformer는 transform 결과로 **자신이 만든 컬럼만 반환**한다(입력 컬럼 포함 X) — downstream에서 concat이 책임
- `get_feature_names_out`이 있으면 `{node}__{col}` 접두사 부여
- 없어도 `output_vars`가 None이고 result에 `.columns`가 있으면 같은 접두사를 fallback으로 적용(`fit_process`/`process` 공통, `process`에선 `self.output_vars` 대신 local 변수로 처리)

## Adapter 인터페이스
- `get_params(params, logger)`: 모델 생성 파라미터
- `get_fit_params(data_dict, params, logger)`: fit 파라미터 — base: X/y를 `unwrap()` 후 반환
- `get_process_data(data)`: `process()` 입력 데이터 변환 — base: `unwrap(data)`
  - `LightGBMAdapter`: polars→pandas 변환 (LightGBM polars 미지원); `early_stopping` dict 수락 → 내부에서 `lgb_early_stopping` 콜백으로 변환 (params에 콜백 인스턴스를 넣을 수 없으므로 이 dict 형태가 유일한 지정 방법)
  - `CatBoostAdapter`: `_catboost_supports_polars()` (>=1.3.0) 기반 분기 — 구버전이면 polars→pandas (`get_fit_params`도 동일 적용)
- `result_objs`: `{name: (callable, mergeable_bool)}`
- **`stack_evals_result(evals_result)`(`_base.py`)**: `{split: {metric: [iteration별 값]}}` → 하나의 stacked Series. XGBoost/LightGBM/CatBoost/NN 네 어댑터의 `_get_evals_result`가 전부 이걸 쓴다
  - **한 split 안에서 metric 곡선 길이가 달라도 된다** — 각 곡선을 `pd.Series`로 감싸므로 iteration 인덱스로 정렬되고 짧은 쪽은 NaN으로 패딩됨. (`pd.DataFrame({metric: list})`로 만들면 `ValueError: All arrays must be of the same length`) 실제 사례: CatBoost `eval_metric='AUC'` + early stopping에서 loss와 AUC의 기록 길이가 어긋남
  - **`.stack()` 뒤의 `.dropna()`는 명시적이어야 한다** — 패딩된 자리를 남길지 떨굴지가 pandas 버전에 따라 달라서, 기본값에 기대면 같은 코드가 환경에 따라 다른 모양을 낸다. 떨구는 쪽으로 고정: CatBoost는 AUC를 validation에만 기록해서 `(iter, 'AUC', 'learn')`이 전 iteration NaN인데, 이게 남으면 `ModelAttrCollector.get_attrs_agg`의 groupby/mean에 섞여 들어간다
  - 빈 `evals_result`는 빈 Series
- `__eq__`: `type(self) is type(other) and self.__dict__ == other.__dict__`
- `__hash__`: `id(self)` — set/dict 키로 사용 가능
- **adapter 지정 방식** (`set_grp`/`set_node`의 `adapter=`): `"module.ClassName"` 문자열 / `{"__ref__": ..., "__params__": {...}}` / `None`만 허용(**인스턴스는 `TypeError`**) — **저장 시점엔 resolve 안 함**, `_node_processor.py`가 인스턴스 생성 시 `resolve_node_adapter(processor, adapter)`로 resolve
  - `resolve_node_adapter(processor, adapter_spec)`: `adapter_spec` 있으면 `resolve_instance(adapter_spec)`, 없으면 `get_adapter(processor)`(processor 클래스명 기반 디폴트 — 문자열이면 `rpartition('.')[-1]`로 bare/`"module.ClassName"` 둘 다 처리, 클래스/인스턴스면 `.__name__`/`.__class__.__name__`)
  - GPU 판정(`need_gpu`)도 이 함수로 resolve — Trial은 `_make_jobs`가, 노드는 `_make_node_jobs`가 job 생성 시점에 이름당 1회 resolve해 `Job.need_gpu`에 박아 넣는다(executor 쪽엔 GPU 판정 캐시가 없다 — job 리스트가 이미 분류돼 있음)
- **레지스트리** (`adapter/__init__.py`): `MODEL_ADAPTERS`(모델명→인스턴스), `get_adapter(model_or_name)`. `NNAdapter`는 TF를 top-level import하므로 **지연 로드** — `_LAZY_ADAPTERS`(`NNClassifier`/`NNRegressor`)로 first-use 시 인스턴스화·캐시, 모듈 `__getattr__`로 `NNAdapter` 심볼 노출 → `import mllabs`가 TF를 끌어오지 않음

## Sampler (`sampler/` 패키지)
- **Sampler** (`_base.py`): 기본 클래스 — `sample(fit_params) → fit_params` 인터페이스
- **ImbLearnSampler** (`_imblearn.py`): imblearn `fit_resample` 래퍼
  - `__init__(sampler)`: imblearn sampler 인스턴스 주입
  - `sample(fit_params)`: `fit_params['X']`/`['y']`로 `fit_resample` 호출 후 X, y 교체하여 반환
- 사용법: node `params`에 `mllab_sampler` 키로 Sampler ref 지정 → `_node_processor`가 fit/fit_process 전에 `sample()` 호출; estimator에 전달 전 키 제거

## 보조 모듈
- **_data_wrapper.py**: DataWrapper (wrap/unwrap/squeeze/mean/mode/simple) — pandas/polars/cudf/numpy 통합
  - **`wrap()`은 멱등** — 이미 `DataWrapper`면 그대로 반환. `unwrap()`이 native에 대해 멱등인 것과 대칭이고, 덕분에 `Experimenter`/`Trainer`가 native든 wrapped든 받을 수 있음
  - `PolarsWrapper.get_columns()`: `pl.DataFrame`이면 `.columns`, `pl.Series`이면 `.name` 반환
  - `select_by_dtype(kind)`: `'category'|'numeric'|'int'|'float'|'str'|'bool'`에 해당하는 컬럼명(numpy는 정수 offset) 리스트 반환 — `col.py`의 `@numeric` 등 dtype selector가 쓰는 primitive
  - 벡터화 원칙: pandas는 `isna()`+`where()`+문자열 벡터 연산(Python 루프 금지), polars는 `map_elements` 금지하고 `when/then/otherwise`+native expression, numpy object 배열은 `np.vectorize`
- **_edge_dsl.py**: edges DSL 파서/평가기 — 위 "Edge DSL" 섹션
- **_serialize.py**: ref 기반 직렬화/해석
  - `serialize_value`/`deserialize_value` (JSON 왕복), `_obj_to_ref`/`_ref_to_obj`
  - `resolve_processor(x)`: `"module.ClassName"` str → 클래스, else passthrough
  - `resolve_instance(spec)`: str→인스턴스(기본값) / `{__ref__, __params__}`→`cls(**params)` / else passthrough. `resolve_adapter`가 위임
  - `resolve_ref_values(value)`: params 값 재귀 해석 — `{"__callable__": "mod.fn"}`→**호출 안 하고** 그 객체 참조(metric_func 등), `{"__ref__": ..., "__params__": {...}}`→인스턴스화, 문자열/스칼라는 그대로. `set_grp`/`set_node`/`set_collector`의 params에 적용
- **_project_store.py**: `ProjectStore` — `{project}/project.db`, `experimenters(name PK)`/`trainers(name PK)` **이름 목록만**. 등록은 `Project` 팩토리가 함 — 직접 생성한 run은 어느 색인에도 안 들어감
- **_experimenter_store.py**: `ExperimenterStore(path)` — **run 하나 전용** `{path}/__exp.db`. meta 행 + splitter BLOB + `save_pipeline(pipeline)`/`load_pipeline()`(`_run_common`에 위임). `fetch`/`load_splitters`/`remove`는 행이 하나뿐이라 `name` 생략 가능
- **_trainer_store.py**: `TrainerStore(path)` — Trainer판 동형. `{path}/__trainer.db`, `trainer(name PK, pipeline_name, pipeline_version, splits BLOB)`. Experimenter판과 다른 점 둘: splits 블롭에 **`split_indices`가 같이** 들어감(splitter가 없을 수도 있고, 학습된 fold와 정확히 같아야 해서), `data_key`/`title`이 **없음**(다른 run과 비교할 일이 없어 라벨도 mismatch 가드도 불필요)
  - **`save(meta)`가 `INSERT OR IGNORE` + `UPDATE`인 이유**(양쪽 store 공통): meta 컬럼만 나열한 `INSERT OR REPLACE`는 같은 행의 BLOB(splitters/splits)을 NULL로 날려버린다
- **_run_common.py**: Experimenter/Trainer 공용 — `require_built_pipeline`, `resolve_common_status`, `save_pipeline(path, pipeline)`/`load_pipeline(path)`(`{path}/pipeline.pkl`, 없으면 `None`)
- **_describer.py**: desc_spec, desc_pipeline, desc_node, compare_nodes
- **_logger.py**: BaseLogger, DefaultLogger (start/update/end_progress, adhoc_progress, rename_progress)
- **col.py / _connector.py / collector/ / filter/ / adapter/ / processor/**: 해당 섹션 참조
- **filter/**: DataFilter, RandomFilter(n/frac/random_state), IndexFilter(index)
- **processor/**: CatConverter, CatPairCombiner, CatOOVFilter, FrequencyEncoder, TypeConverter, CrossFitTransformer (`ColSelector`는 `_pipeline.py` 소속)
  - `CatPairCombiner`: pair(2) → N-way 그룹 조합으로 확장. `pairs` 요소를 N개 컬럼 인덱스/이름 그룹으로 지정 가능
  - `TypeConverter`: 모든 컬럼을 지정 타입(`str`/`int`/`float`)으로 변환. pandas: `astype`, polars: cast, numpy: `astype`. `get_feature_names_out` 지원
  - `CrossFitTransformer`: sklearn-compatible stacking meta-feature 생성기
    - `__init__(estimator, cv=5, method='predict_proba', stratified=True)`
    - `fit_transform`: CV로 OOF 예측 생성 + 전체 데이터로 full estimator fit
    - `transform`: full estimator로 예측 (fit_transform 이후)
    - 출력 컬럼명: `{estimator_class_lower}_{class}` (predict_proba) / `{estimator_class_lower}_pred` (predict)
    - 노드로 사용 시 Experimenter는 OOF, Trainer/Inferencer는 full model 경로로 동작
  - polars 설치 시: PolarsLoader, ExprProcessor, PandasConverter 추가
  - `_dproc.py`: `get_type_df` (수치형만 f32/i32/i16/i8 판정), `get_type_pl`, `get_type_pd`, `merge_type_df`

## 실행 (`_executor.py`, `_tracker.py`)

### Job
`Job(name, spec, outer_idx, inner_idx, flow, need_gpu=False)` — 노드와 Trial/Predictor 공용 job 단위.
`spec`은 `Pipeline.get_node_spec()`/`Trial.get_spec()`/`Predictor.get_spec()`이 준 `ProcessorSpec`을 job 생성 시점에 1회 계산해 박아 넣은 것(따로 `node`/`trial` 객체를 들고 있지 않음). `flow`(`TrainDataFlow`) 하나로 `get_train`/`get_valid`/`get_test(edges)`를 다 만들 수 있어 job이 자족적이다. 결과가 **어디로 가는지는 job이 아니라 executor의 `store`**가 정한다 — job의 flow가 읽는 store와 같으란 법이 없다.

### `_execute_single(jobs, store, gpu_id_list=None, collectors=None, tracker=None, chained=False)`
단일 프로세스로 `Job` 리스트를 실행. 세 종류가 전부 같은 `_process`를 타고, 호출부가 바꾸는 건 **결과가 어디로 가느냐**뿐이다.

- **`store`는 그 job 종류의 기록을 소유한 store** — 노드/Predictor는 `NodeStore`, Trial은 `TrialStore`. 아티팩트 기록은 `store.stores_artifacts`가 True일 때만(Trial은 False라 아무것도 안 남는다)
- **`chained`는 "이 job들이 flow를 통해 서로 먹이는가"** — Pipeline 노드에서만 True. 두 가지가 같이 켜진다: 완료된 obj/result를 `set_objs`로 flow에 게시(뒤 job이 읽으라고), 그리고 edges가 참조하는 게 다 빌드될 때까지 대기(`get_missing_nodes`). leaf(Trial/Predictor)는 둘 다 안 함
  - 대기 게이트를 leaf에 걸면 안 되는 이유: 참조 노드가 끝내 안 빌드되면 그 job이 에러 하나 없이 조용히 사라진다 — `_job_inputs`가 `KeyError`를 내고 prep error로 기록되는 게 올바른 동작
  - `set_objs`를 leaf에 하면 안 되는 이유: 아무도 안 읽는데 학습된 모델만 flow 메모리에 실행 내내 붙어 있게 된다
- **완료 표시는 `done` 집합**(`(outer_idx, inner_idx, name)`) — 예전엔 `flow.node_objs` 등록 여부가 겸했는데, 그러면 flow에 안 올리는 leaf가 영원히 `ready`로 남는다
- **`collectors`는 별개**로 Collector 얘기만 한다: `None`이면 `ext_data` 준비와 매칭을 통째로 스킵(노드엔 Collector가 안 붙음), 리스트면 실행(`[]`는 매치될 게 없는 리스트일 뿐)
- **반환값은 job 종류와 무관하게 항상 `{(outer_idx, inner_idx, name): error_info}`** — 실패한 job이 ready-루프에서 영원히 재시도되지 않으려면 키가 fold까지 포함한 job 신원이어야 하고, 반환도 그 신원 그대로여야 한다(축약하면 같은 outer fold의 다른 inner fold 실패가 사라지고, 호출부의 `len(jobs) - len(errors)` 집계도 틀어짐). 호출부는 이름만 필요할 때 `{n for _, _, n in errors}`로 뽑는다

**호출부별 조합**: `build()` = `node_store` + `chained=True` / `exp()` = `trial_store`(저장 없음) + collectors / `train()` 노드 = `node_store` + `chained=True` / `train()` Predictor = `predictor_store` + `collectors=[]`

### `_execute_multi(jobs, n_jobs, store, gpu_id_list=None, collectors=None, tracker=None, ..., chained=False)`
워커 풀 실행. `store`/`chained`/`collectors`의 의미와 반환 모양은 `_execute_single`과 동일.
- **`chained`일 때만 완료 결과를 되읽어 flow에 올린다** — 되읽기는 `job.flow`가 아니라 인자로 받은 `store`에서. 워커는 이 호출이 지정한 store에 쓰기 때문이고, chained면 둘이 어차피 같지만(노드의 flow는 자기가 쓴 store를 읽는다) 그 전제를 코드가 기대는 대신 말하게 둔 것
- **ready-job 계산은 매 dispatch 사이클마다 처음부터 다시 스캔**(`_collect_ready()`) — 노드는 형제 노드가 끝나야 readiness가 바뀌므로 목록을 한 번만 만들어두는 방식이 안 맞는다. leaf에도 그대로 맞음(서로 의존 안 하니 한 번 ready면 계속 ready)
- **워커 배정 fallback**: "내 타입" job이 아직 남아있으면 그 타입 몫 worker를 다른 타입에 안 뺏긴다(`elif free_cpu and not cpu_ready and gpu_fallback_cpu`). ready 목록을 매 사이클 재계산하는 탓에 같은 `_try_dispatch()` 호출 안에서 GPU pass가 막 dispatch한 job이 CPU pass의 판정엔 반영 안 되지만(다음 'done'/'error' 이벤트에서 바로잡힘) 무시할 수준
- `ProcessWorker(conn, collectors or [], store, ...)`로 store를 그대로 넘김 — 워커 메시지 튜플은 `spec, outer_idx, inner_idx, train_data, valid_data, test_data, ext_data`(워커가 `store.write_objs(spec.name, ...)`로 직접 쓰므로 경로를 미리 조립해 보낼 필요 없음)
- `ProcessWorker`(spawn): job 경계에서 `del` + `gc.collect()`로 이전 job의 데이터·모델을 놓아줌(안 하면 피크 = 이전 데이터 + 모델 + 새 데이터). 워커 로그 fd는 dup2 직후 close

#### 워커 사망 처리
`wait()`는 파이프가 닫힌 것도 ready로 돌려주므로, OOM kill이나 네이티브 라이브러리 segfault로 워커가 죽으면 `recv()`가 `EOFError`를 던진다. 이걸 예외로 전파시키면 정리 코드에 도달하지 못해 나머지 워커가 종료 sentinel을 못 받고 영원히 블록된다(주피터 커널은 안 죽으므로 `daemon=True`도 소용없어 메모리와 CUDA 컨텍스트를 잡은 채 남는다).
- 사망을 **예외가 아니라 정상 결과**로 취급: in-flight job을 `WorkerLost` 에러로 기록하고, 그 conn을 `all_conns`에서 제거(EOF는 지속 상태라 안 빼면 무한 스핀), 남은 워커로 계속 진행. 전멸하면 못 돌린 job까지 에러로 남겨 호출부 집계가 안 틀리게 함
- **정리는 `try/finally`** — 루프를 빠져나가는 경로가 여럿(`store.get_objs`, 이력 SQLite 쓰기, tracker 호출)이라 "끝까지 도달했나"에 정리를 걸면 안 된다. `send(None)` → `join(_JOIN_TIMEOUT=10)` → 살아있으면 `terminate()` → `close()`, 각 단계 예외 안전
- 기동 시 `'ready'` 대기도 같은 처리 — spawn은 모듈을 재import하므로 Collector 클래스가 모듈 최상위가 아니면 자식이 `'ready'` 전에 죽는다
- **미해결**: 루프 안 부모쪽 작업(`store.get_objs`/`flow.set_objs`/`tracker.*`/`abort_node`)의 개별 예외는 여전히 실행 전체를 중단시킨다. `finally` 덕에 누수는 없지만 job 단위로 흡수하지는 않음

#### Collector 실행
- **`_run_collectors`는 `(warn_msgs, outcomes)`를 반환** — `outcomes`는 매칭된 Collector당 하나씩 `{collector, status, elapsed, info}`. 실제 캐칭은 `_safe_collect`가 `ext`/`collect`/`push` 세 구간에 대해 하고, 공용 `obj.process` 준비(`output_test`/`output_train`)가 깨지면 매칭된 전원에게 `phase='output'` outcome을 발급한다 — 이 준비는 Trial이 이미 `'built'`로 기록된 **뒤에** 돌기 때문에 여기서 예외가 새면 실행 전체가 죽는다
- 멀티워커: 워커가 결과는 `('collect', ...)` 메시지로, outcome은 `('collect_hist', node, o, i, outcomes)` 메시지로 보낸다(같은 파이프라 순서 보장). 부모는 `('collect', ...)` 처리에서 `c.push()`를 try로 감싸 실패를 `push_errors`에 담아뒀다가 `collect_hist` 도착 시 해당 outcome을 교체해 기록. **결과가 picklable하지 않아 `('collect', ...)` send 자체가 실패해도** `_safe_collect`가 `phase='push'`로 잡고, outcome은 문자열뿐이라 따로 전달된다
- 기록은 항상 부모(`tracker.collect(...)` → `TrialHistTracker`) — 워커가 SQLite를 직접 쓰면 N프로세스 경합이 생긴다

### _tracker.py
- `ExecuteTracker` 기반. `LoggerExecuteTracker` — 워커 이벤트→logger, `typ`에 따라 `logger.info`/`warning` 라우팅
- `NodeInfoTracker` — 노드/Predictor 실행 이력을 그 run의 `NodeStore.node_hist`에 기록
- **`TrialHistTracker(tracker, store, experimenter, pipeline_version, collect_hist=None)`** — 로깅 tracker를 감싸 `done`/`error` 시점에 `TrialStore`에 이력 기록. 이벤트 시점이라 멀티워커도 그대로 커버되고, 사후에 디스크를 다시 읽지 않아도 된다
  - `collect_hist`(실행 중인 Experimenter의 `collectors.hist`)를 주면 `collect(node_name, outer_idx, inner_idx, outcomes)` 이벤트에서 `CollectHist`에도 기록. 별도 wrapper 클래스를 두지 않은 이유: 스탬프가 이 클래스가 이미 들고 있는 `pipeline_version` 그대로고, 부모 프로세스 이벤트 스트림을 타야 하는 이유도 같다. `experimenter`는 안 넘어간다 — 그 hist가 이미 그 run의 것
  - `ExecuteTracker.collect(...)`는 base에 no-op — 노드/Predictor 경로엔 매칭될 Collector가 없어 실제로 호출되지 않음

## 저장 구조
**Project가 경로를 소유한다.**
```
{project.path}/
  project.db                        # ProjectStore — experimenters(name PK) / trainers(name PK).
                                    # 이름 목록만. run에 대한 정보는 전부 run 디렉토리에
  trials.db                         # trials + experiment_hist

  pipelines/{name}/
    {name}.db                       # PipelineBuilder 노드/그룹 정의 + versions(version PK, path)
    v{n}.pkl                        # 버전별 빌드 결과 Pipeline

  exp/{name}/                       # Experimenter — 이름이 곧 식별자, 디렉토리 하나로 자족
    __exp.db                        # ExperimenterStore — 이 run 전용.
                                    #   experimenter(name PK, data_key, title,
                                    #                pipeline_name, pipeline_version, splitters BLOB)
    pipeline.pkl                    # 이 run이 채택한 Pipeline 사본
    collectors/                     # 이 run의 Collectors — 프로젝트 전역 레지스트리는 없다.
                                    #   Collector 데이터가 노드 이름만으로 키잉되므로
                                    #   경로가 run을 가르는 유일한 수단
      collectors.db                 #   CollectorStore — collectors(name PK, collector,
                                    #     connector, path). 정의의 평문 절반
      __params/{name}.pkl           #   정의의 나머지 절반 — 생성자 params (산 객체가
                                    #     들어올 수 있어 pickle). 이 둘로 재조립
      collect_hist.db               #   CollectHist — 정의(위)와 별도 파일로,
                                    #     CollectorStore는 정의만 갖는다는 경계 유지
      {name}/                       #   Collector가 소유하는 저장 위치 — 데이터만
        metrics.db                  #     MetricCollector (node, idx, inner_idx, split, value)
        {node}.pkl                  #     StackingCollector — {'folds': [[inner 결과...], ...]}
        {node}/{idx}_{inner_idx}.pkl  #   OutputCollector
    __worker_logs/worker_{i}.log    # 멀티워커가 캡처한 네이티브 출력 (+ master.log)
    __folds/__node_hist.db          # 이 run의 NodeStore가 소유
    __folds/{outer_idx}/{inner_idx}/{name}/
      obj.pkl                       # processor 객체
      result.pkl                    # fit_transform/fit_predict 출력
      # 여기 있는 건 Pipeline 노드뿐이다 — Trial은 아무것도 안 남긴다.
      # info 파일도 없다: status/definition/edges 등은 전부
      # __node_hist.db(노드) / trials.db의 experiment_hist(Trial)에
      # 있고, NodeStore 자신은 obj.pkl 존재만 안다

  trainers/{name}/
    __trainer.db                    # TrainerStore — 이 Trainer 전용.
                                    #   trainer(name PK, pipeline_name, pipeline_version,
                                    #           splits BLOB = splitter/splitter_params/split_indices)
    pipeline.pkl                    # 이 Trainer가 채택한 Pipeline 사본
    __node_hist.db                  # 노드용 NodeStore — exp/{name}과 별개 base path
    {split_idx}/0/{name}/           # 노드 obj.pkl / result.pkl
    __predictors/                   # Predictor는 별도 디렉토리
      __node_hist.db                #   Predictor용 NodeStore — 노드 것과 파일명이 같아 분리가 강제됨
      __predictors.db               #   PredictorStore (정의만)
      {split_idx}/0/{name}/         #   Predictor obj.pkl / result.pkl

  inferencers/{name}/
    __inferencer.pkl                # node_specs, selected_nodes/predictors, n_splits, node_objs, v
```
- **Experimenter/Trainer는 각자 `pipeline.pkl` 사본을 소유한다** — 자기 디렉토리만으로 재개 가능한 것이 목적. `(pipeline_name, pipeline_version)`은 **provenance**로만 남는다(이 사본이 프로젝트의 어느 버전에서 왔는지)
  - 실제 I/O는 `_run_common.save_pipeline`/`load_pipeline` 한 곳. Experimenter는 `ExperimenterStore`를 통해, Trainer는 직접 호출
  - `Project.load_experimenter`/`load_trainer`는 **버전을 resolve하지 않는다** — run이 자기 사본을 읽는다. 버전으로 지정하는 건 생성 팩토리(`experimenter()`/`trainer()`의 `pipeline_version=`)뿐
- **NodeStore는 project 전역이 아니라 run 하나당 하나** — 프로젝트 전역 노드 이력 레지스트리는 없다. `Experimenter.node_store`/`Trainer.node_store`가 생성자에서 자기 base path로 만들어 모든 fold가 공유

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
