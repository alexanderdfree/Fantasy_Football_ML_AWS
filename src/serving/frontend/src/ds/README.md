# Design-system component kit

This directory retains the React primitives supplied with the dashboard's design
system. Only [controls/DropdownMenu.jsx](controls/DropdownMenu.jsx) is currently
imported by the app, from [NextWeek.jsx](../views/NextWeek.jsx) and
[SeasonLeaders.jsx](../views/SeasonLeaders.jsx). The other primitives are dormant
kit assets: they are not reachable from `main.jsx` and esbuild excludes them from
the runtime bundle.

For changes to the live UI, start in the importing view. Shared components such
as `PillGroup`, `PlayerCell`, and `Pagination` live in
[components/common.jsx](../components/common.jsx); their props differ from the
similarly named kit components here. Editing a dormant component does not change
the dashboard. Adopting more of this kit would be a separate refactor.

The design-system CSS tokens and themes are used by the live app independently
of these component imports. See
[ADR-0023](../../../../../docs/adr/0023-react-frontend-esbuild-committed-bundle.md)
for the source and committed-bundle workflow.
