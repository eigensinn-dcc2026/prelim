import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {finalizePresentation} from 'file:///C:/Users/sooyah/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const skill = 'C:/Users/sooyah/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const result = await finalizePresentation({workspaceDir:root,candidatePath:path.join(root,'outputs/submission_deck/candidate_repaired.pptx'),finalPath:path.join(root,'submission/DCC2026_예선_결과정리.pptx'),pythonExecutable:'C:/Users/sooyah/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),explicitTotalSlideCount:19,requiredNativeTableOwnerSlides:[5,13,16,17],requiredNativeChartOwnerSlides:[10,12,15],layoutArgs:[...[5,13,16,17].flatMap(n=>['--require-native-table-slide',String(n)]),'--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],verifyArtifactToolImport:true,receiptPath:path.join(root,'outputs/submission_deck/validation.json')});
console.log(JSON.stringify(result));
