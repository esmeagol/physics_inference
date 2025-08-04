# Requirements Document

## Introduction

The tracking system has inconsistent field naming for object class information. Some parts use `'class'`, others use `'class_name'`, causing class information to be lost and show as 'N/A' in outputs. We need to standardize on using `'class'` everywhere.

## Requirements

### Requirement 1

**User Story:** As a developer, I want consistent class field naming so that class information flows correctly through the tracking pipeline.

#### Acceptance Criteria

1. WHEN tracker results are generated THEN they SHALL use `'class'` field for object class information
2. WHEN analysis tools process results THEN they SHALL read from `'class'` field only
3. WHEN tracking analysis is generated THEN it SHALL display actual class names instead of 'N/A'

### Requirement 2

**User Story:** As a developer, I want all `'class_name'` references removed so there's no field name confusion.

#### Acceptance Criteria

1. WHEN searching the codebase THEN there SHALL be no `'class_name'` field lookups remaining
2. WHEN tracker implementations return results THEN they SHALL use `'class'` as the field name
3. WHEN the changes are complete THEN all fallback logic for `'class_name'` SHALL be removed