// @hey-api/openapi-ts generates code through TypeScript's JS compiler API
// (ts.SyntaxKind etc.), which typescript@7 (the native Go port) does not ship.
// Its peer range (>=5.5.3) still semver-matches 7.x, so pnpm would link it
// against the project's TS 7 and the generator crashes on import. Give it a
// private TS 5.x instead; the project itself stays on TS 7.
// Do not remove until openapi-ts supports the TS 7 API surface.
function readPackage(pkg) {
  if (pkg.name === "@hey-api/openapi-ts") {
    delete pkg.peerDependencies.typescript;
    pkg.dependencies = { ...pkg.dependencies, typescript: "~5.9.0" };
  }
  return pkg;
}

module.exports = { hooks: { readPackage } };
