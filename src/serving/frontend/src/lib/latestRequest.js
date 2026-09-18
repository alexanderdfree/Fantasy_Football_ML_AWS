/* Keep async view updates tied to the latest navigation, including cache hits. */
export function createLatestRequest() {
    let current = 0;
    return {
        async run(load, onSuccess, onError) {
            const request = ++current;
            try {
                const value = await load();
                if (request === current) onSuccess(value);
            } catch (error) {
                if (request === current) onError(error);
            }
        },
        cancel() { current += 1; },
    };
}
