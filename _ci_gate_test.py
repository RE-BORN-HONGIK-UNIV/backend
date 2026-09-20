# 일부러 정의 안 된 이름을 써서 ruff(F821)에 걸리게 한 테스트용 파일 —
# branch protection(status check 게이팅)이 실제로 Merge 버튼을 막는지
# 검증하기 위한 것. 확인 끝나면 PR과 함께 정리함.
result = this_name_is_never_defined + 1
