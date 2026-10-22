export async function main(device) {
  const info = device.adapterInfo ?? {};
  const l = device.limits;
  return { vendor: info.vendor, arch: info.architecture, desc: info.description, features: [...device.features],
    maxStorageBufferBindingSize: l.maxStorageBufferBindingSize, maxBufferSize: l.maxBufferSize,
    maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
    maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage };
}
