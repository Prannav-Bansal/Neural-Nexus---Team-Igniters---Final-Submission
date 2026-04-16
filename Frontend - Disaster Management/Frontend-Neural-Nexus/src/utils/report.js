const currencyFormatter = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
});

export function deriveInsights(result) {
  const severityMap = {
    low: { damagePct: 28, lossBase: 90000, risk: 'Low', authenticity: 'Likely Genuine' },
    medium: { damagePct: 57, lossBase: 260000, risk: 'Medium', authenticity: 'Possibly Exaggerated' },
    high: { damagePct: 81, lossBase: 620000, risk: 'High', authenticity: 'Suspicious Claim' },
  };

  const severityKey = (result.damageSeverity || 'medium').toLowerCase();
  const profile = severityMap[severityKey] || severityMap.medium;
  const confidence = Number(result.confidenceScore || 0.78);
  const metadataBoost = result.metadata?.damageSpread ? 0.08 : 0;
  const damagePct = Math.min(95, Math.round((profile.damagePct + confidence * 10 + metadataBoost * 100) * 10) / 10);
  const lowerLoss = Math.round(profile.lossBase * (0.84 + confidence * 0.06));
  const upperLoss = Math.round(profile.lossBase * (1.08 + confidence * 0.08));

  let authenticity = profile.authenticity;
  if (confidence >= 0.9 && severityKey !== 'high') authenticity = 'Likely Genuine';
  if (confidence < 0.68 && severityKey === 'high') authenticity = 'Suspicious Claim';

  return {
    damagePct,
    estimatedLoss: `${currencyFormatter.format(lowerLoss)} - ${currencyFormatter.format(upperLoss)}`,
    riskScore: profile.risk,
    authenticity,
  };
}

export function buildReport({ formData, result, insights }) {
  return [
    'DISASTER DAMAGE INTELLIGENCE REPORT',
    'AI-Powered Insurance Claim Verification',
    '',
    `Claim ID: ${formData.claimId || 'Not provided'}`,
    `Location: ${formData.location || 'Not provided'}`,
    `Incident Type: ${formData.incidentType || 'Auto-detected'}`,
    '',
    `Disaster Type: ${result.disasterType}`,
    `Damage Severity: ${result.damageSeverity}`,
    `Confidence Score: ${Math.round((result.confidenceScore || 0) * 100)}%`,
    '',
    `Estimated Damage: ${insights.damagePct}%`,
    `Estimated Loss Range: ${insights.estimatedLoss}`,
    `Risk Score: ${insights.riskScore}`,
    `Claim Authenticity Check: ${insights.authenticity}`,
    '',
    'Metadata Summary:',
    JSON.stringify(result.metadata || {}, null, 2),
  ].join('\n');
}
