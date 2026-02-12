# TODO

## ~~1. 파일럿 노트북 실행 & 검증~~ (완료)
- [x] `1. Modeling.ipynb` XGB/CatBoost 섹션 실제 실행
- [x] adapter별 `evals_result` 키 이름 검증
- [x] 결과 비교 → 사용성 개선 필요 (→ 4번으로)

## ~~2. Stacking 워크플로우 완성~~ (완료)
- [x] `_stacking.py`(레거시) 제거, `collector/_stacking.py`로 통일
- [x] `include_target=False`일 때 불필요한 target 빌드 제거
- [x] 파일럿 노트북에서 StackingCollector로 LGB+XGB+CB 결과 수집
- [x] 2nd level 모델: get_dataset → 새 Experimenter 생성

## 4. 프레임워크 사용성 개선 (Frozen → ml-lab 분리 후 진행)
- [x] Node error 상태 구현 (build/exp 중 에러 시 error 상태 전환, 나머지 노드 계속 진행)
- [x] Experimenter 상태 요약 (`desc_status`: Stage/Head 상태 통계 + 에러 상세)
- [ ] 에러 분석 도구 (오분류 케이스 분석)
- [ ] 보일러플레이트/템플릿 기능 (반복되는 Experimenter 셋업 코드 간소화)

## ~~5. 패키지화 준비~~ (완료)
- [x] 패키지 이름 결정 → `ml-lab` (import: `mllab`)
- [x] `pyproject.toml` 구성
- [x] `__init__.py` 정리, 의존성 명시
- [x] PolyForm Noncommercial 1.0.0 라이선스 적용

## 6. 새로운 Collector 타입 (Frozen → ml-lab 분리 후 진행)
- [x] OutputCollector (output_train/output_valid 저장, ConfusionMatrix 등 사후 분석 커버)
- ~~ConfusionMatrixCollector~~ → OutputCollector에서 사후 계산으로 대체
- [ ] ObjVarsCollector (fold별 입력/출력 변수 구성 수집, finalize 후에도 변수 정보 보존 → Report 정확성 확보)
- [ ] CalibrationCollector

## 7. PGS5_ep11 마무리
- [x] `processor/` → `mllab/`에 병합
- [X] Train (모델 학습)
- [X] Inference (테스트셋 예측 결과 생성)
- [ ] 최종 아웃풋 노트북 작성
